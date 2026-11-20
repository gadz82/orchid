"""LangGraph topology for a compiled pipeline, and the job-node algorithm.

This is the only module that knows the graph shape: a reserved start node, one
node per job, a synthetic join node for ``join: any``, conditional edges for a
decision's branches, and a finish node. The graph is compiled **without** a
checkpointer — pipeline runs are not resumable by design (spec §7.7.2).

Each job node is the same callable with the job bound: it reads the run session
from ``RunnableConfig`` (never from state), applies the ``when`` gate, executes
the job's step with retry/timeout, turns the step result and the declared
writes/links into mutations, commits them through the session, and records the
job result. A normal step failure never raises out of the node: it is recorded
and the graph keeps going so the final status and partial results survive.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import END, START, StateGraph

from ..config.tool_registry import call_tool
from ..core.pipeline import (
    OrchidGraphMutation,
    OrchidGraphObject,
    OrchidGraphRelation,
    OrchidPipelineAuthError,
    OrchidPipelineConfigError,
    OrchidPipelineError,
    OrchidPipelineInputError,
    OrchidPipelineJobContext,
    OrchidPipelineMappingError,
    OrchidPipelineRetryableError,
    OrchidPipelineSerializationError,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
    _jsonable,
)
from ..core.run_config import auth_from_config
from .definition import CompiledJob, CompiledPipeline, PipelineDependencies
from .mapping import evaluate, evaluate_when, render_template, render_value
from .registry import PIPELINE_STEP_REGISTRY, resolve_pipeline_step
from .session import PipelineJobResult, PipelineRunSession, PipelineStepContext
from .steps import register_builtin_steps
from .steps.map import MapNodes, MapRuntime, build_map_nodes

__all__ = [
    "PipelineExecution",
    "PipelineGraphState",
    "build_pipeline_graph",
    "final_status",
    "pipeline_session_from_config",
]

_START_NODE = "__pipeline_start__"
_FINISH_NODE = "__finish__"
_JOIN_PREFIX = "__join__"
_MAX_BACKOFF_SECONDS = 300.0


class PipelineGraphState(TypedDict, total=False):
    """The small, JSON-safe state the pipeline graph carries.

    The information graph lives in the run session, not here, so partial
    results survive a failure and no object data enters a checkpoint channel.
    Map plans and item records are bounded by ``max_map_items``; they carry the
    operator's collection/items, not graph objects.
    """

    run: dict[str, Any]
    input: dict[str, Any]
    seed: list[dict[str, Any]]
    job_results: Annotated[dict[str, Any], _merge_job_results]
    map_plan: Annotated[dict[str, Any], _merge_job_results]
    map_header: Annotated[dict[str, Any], _merge_job_results]
    map_items: Annotated[dict[str, Any], _merge_job_results]
    errors: Annotated[list[dict[str, Any]], _extend_errors]
    status: str
    output: dict[str, Any] | None
    summary: str
    error: str | None


def _merge_job_results(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged


def _extend_errors(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*left, *right]


@dataclass(frozen=True, slots=True)
class PipelineExecution:
    """Execution-time collaborators the job nodes receive.

    Build-time references (agents, tools, decision models) live in
    :class:`PipelineDependencies`; this bag holds the runtime objects a step may
    use. None of these are serialized: the session and this bag travel in
    ``RunnableConfig``.
    """

    chat_model: Any | None = None
    reader: Any | None = None
    signal_emitter: Any | None = None
    tool_caller: Callable[..., Any] | None = None
    child_runner: Any | None = None
    logger: logging.Logger | None = None


def pipeline_session_from_config(config: RunnableConfig | None) -> PipelineRunSession:
    """Read the run session from ``config.configurable``; a missing one is fatal."""
    configurable = (config or {}).get("configurable") or {}
    session = configurable.get("pipeline_session")
    if not isinstance(session, PipelineRunSession):
        raise OrchidPipelineConfigError("pipeline graph invoked without a run session in RunnableConfig")
    return session


# ORCHID-MEM: 01a1ed8b-2401-72bc-ad10-fb55d494ecda — Pipeline graph builder; start/job/join/finish topology, no checkpointer, branch routing and job-node execution.
def build_pipeline_graph(
    definition: CompiledPipeline,
    *,
    deps: PipelineDependencies | None = None,
    execution: PipelineExecution | None = None,
) -> Any:
    """Build and cache the compiled graph for ``definition``.

    The topology is complete for every kind the configuration schema declares;
    a kind whose step module has not been imported fails here with a
    configuration error rather than at run time.
    """
    if not isinstance(definition, CompiledPipeline):
        raise TypeError("build_pipeline_graph requires a CompiledPipeline")
    if definition.graph is not None:
        return definition.graph

    register_builtin_steps()
    _validate_kinds(definition)
    _validate_conditional_targets(definition)
    template_ids = _map_template_ids(definition)
    _validate_map_topology(definition, template_ids)
    execution = execution or PipelineExecution()
    map_runtime = MapRuntime(
        execute_with_retry=_execute_with_retry,
        build_mutations=_build_mutations,
        record_skip=_record_skip,
        record_failure=_record_failure,
        record_terminal=_record_terminal,
        resolve_step=_resolve_step,
        usage_int=_usage_int,
        session_from_config=pipeline_session_from_config,
    )

    map_nodes: dict[str, MapNodes] = {}
    graph = StateGraph(PipelineGraphState)
    graph.add_node(_START_NODE, _make_start_node(definition))
    for job in definition.ordered_jobs:
        if job.id in template_ids:
            # A map template is executed only by its item node.
            continue
        if job.kind == "map":
            nodes = build_map_nodes(definition, job, deps, execution, map_runtime)
            map_nodes[job.id] = nodes
            graph.add_node(job.id, nodes.coordinator)
            graph.add_node(nodes.item_node, nodes.item)
            graph.add_node(nodes.join_node, nodes.join)
        else:
            graph.add_node(job.id, _make_job_node(definition, job, deps, execution))
    graph.add_node(_FINISH_NODE, _make_finish_node())
    graph.add_edge(START, _START_NODE)

    dependents: dict[str, list[str]] = {job_id: [] for job_id in definition.jobs if job_id not in template_ids}
    for job in definition.ordered_jobs:
        if job.id in template_ids:
            continue
        for dependency in job.depends_on:
            dependents[dependency].append(job.id)

    # A conditional target (branch target or job-id fallback) is entered by its
    # decision's conditional edge only: it must not be a root and must not take
    # a normal dependency edge, or it would run even when another branch wins.
    conditional_targets = _conditional_targets(definition)

    dependents_of_decision: dict[str, set[str]] = {}
    for job in definition.ordered_jobs:
        if job.kind == "decision":
            targets = set(job.branches.values())
            fallback_job = _fallback_job_id(job, definition)
            if fallback_job is not None:
                targets.add(fallback_job)
            dependents_of_decision[job.id] = targets

    def exit_node(job_id: str) -> str:
        """Where a job's completion is observed by its dependents."""
        return map_nodes[job_id].join_node if job_id in map_nodes else job_id

    for job in definition.ordered_jobs:
        if job.id in template_ids:
            continue
        if not job.depends_on and job.id not in conditional_targets:
            graph.add_edge(_START_NODE, job.id)

        if job.join == "any":
            join_node = f"{_JOIN_PREFIX}{job.id}"
            graph.add_node(join_node, _make_join_node())
            for dependency in job.depends_on:
                graph.add_edge(exit_node(dependency), join_node)
            graph.add_conditional_edges(join_node, _make_join_router(job), {"run": job.id, "skip": END})
        else:
            for dependency in job.depends_on:
                if job.id in conditional_targets and dependency in dependents_of_decision:
                    # The decision's conditional edge is the only entry point.
                    continue
                graph.add_edge(exit_node(dependency), job.id)

        if job.kind == "map":
            nodes = map_nodes[job.id]
            graph.add_conditional_edges(
                job.id,
                nodes.router,
                [nodes.item_node, nodes.join_node],
            )
            graph.add_edge(nodes.item_node, nodes.join_node)

        if job.kind == "decision" and (job.branches or _fallback_job_id(job, definition)):
            branch_targets = set(dependents_of_decision.get(job.id, set()))
            graph.add_conditional_edges(
                job.id,
                _make_decision_router(job),
                {target: target for target in branch_targets if target in definition.jobs} | {_FINISH_NODE: END},
            )
        else:
            branch_targets = set()

        normal_dependents = [dependent for dependent in dependents[job.id] if dependent not in branch_targets]
        if not normal_dependents and not branch_targets:
            graph.add_edge(exit_node(job.id), _FINISH_NODE)

    graph.add_edge(_FINISH_NODE, END)
    compiled = graph.compile()
    definition.set_graph(compiled)
    return compiled


def _fallback_job_id(job: CompiledJob, definition: CompiledPipeline) -> str | None:
    """Return the job-id fallback, or ``None`` when the fallback is a label/none.

    A declared label wins over a job id with the same name, matching the
    configuration rule and the decision router.
    """
    if job.fallback is None or job.fallback in job.branches:
        return None
    labels = set(getattr(job.config, "labels", ()) or ())
    if job.fallback in labels:
        return None
    return job.fallback if job.fallback in definition.jobs else None


def _conditional_targets(definition: CompiledPipeline) -> set[str]:
    """Every job entered by a decision's conditional edge."""
    targets: set[str] = set()
    for job in definition.ordered_jobs:
        if job.kind != "decision":
            continue
        targets.update(job.branches.values())
        fallback_job = _fallback_job_id(job, definition)
        if fallback_job is not None:
            targets.add(fallback_job)
    return targets


# ORCHID-MEM: 01a1f2c5-f5d6-7d23-8648-529a923bc4ce — Map topology; template jobs run only through item nodes and the join feeds dependents.
def _map_template_ids(definition: CompiledPipeline) -> set[str]:
    """Template job ids that map jobs execute per item instead of as nodes."""
    template_ids: set[str] = set()
    for job in definition.ordered_jobs:
        if job.kind != "map":
            continue
        if not job.map_template:
            raise OrchidPipelineConfigError(f"pipeline {definition.name!r} map job {job.id!r} has no template job")
        template = definition.jobs.get(job.map_template)
        if template is None:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} map job {job.id!r} references unknown template job {job.map_template!r}"
            )
        if template.kind == "map":
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} map job {job.id!r} maps over another map job; "
                "nested maps are not supported"
            )
        if template.kind == "decision" and template.branches:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} map job {job.id!r} maps over a branching decision job"
            )
        template_ids.add(job.map_template)
    return template_ids


def _validate_map_topology(definition: CompiledPipeline, template_ids: set[str]) -> None:
    """Template jobs are not normal nodes; reject edges that would need one."""
    for job in definition.ordered_jobs:
        if job.kind == "map":
            for generated in (f"{job.id}__item", f"{job.id}__join"):
                if generated in definition.jobs:
                    raise OrchidPipelineConfigError(
                        f"pipeline {definition.name!r} map job {job.id!r} generated node name "
                        f"{generated!r} collides with a declared job"
                    )
        for dependency in job.depends_on:
            if dependency in template_ids:
                raise OrchidPipelineConfigError(
                    f"pipeline {definition.name!r} job {job.id!r} depends on map template {dependency!r}; "
                    "template jobs run only through their map item"
                )
    for template_id in template_ids:
        template = definition.jobs.get(template_id)
        if template is not None and template.depends_on:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} map template {template_id!r} must not declare dependencies; "
                "move them to the map job"
            )
    for target in _conditional_targets(definition):
        if target in template_ids:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} conditional target {target!r} is a map template; "
                "target the map job instead"
            )


def _validate_conditional_targets(definition: CompiledPipeline) -> None:
    """A conditional target may depend only on its decision (spec §7.9.2)."""
    targets = _conditional_targets(definition)
    decisions_by_target: dict[str, set[str]] = {target: set() for target in targets}
    for job in definition.ordered_jobs:
        if job.kind != "decision":
            continue
        targets_for_job = list(job.branches.values())
        fallback_job = _fallback_job_id(job, definition)
        if fallback_job is not None:
            targets_for_job.append(fallback_job)
        for target in targets_for_job:
            if target in decisions_by_target:
                decisions_by_target[target].add(job.id)
    for job in definition.ordered_jobs:
        if job.id not in targets:
            continue
        allowed = decisions_by_target.get(job.id, set())
        unexpected = [dependency for dependency in job.depends_on if dependency not in allowed]
        if unexpected:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} job {job.id!r} is a decision target and must depend only on "
                f"its decision job; unexpected dependencies: {', '.join(sorted(unexpected))}"
            )


def _validate_kinds(definition: CompiledPipeline) -> None:
    for job in definition.ordered_jobs:
        if job.kind == "map":
            # Map jobs are topology, not a registered step.
            continue
        if isinstance(job.handler, OrchidPipelineStep):
            continue
        if callable(job.handler) and job.kind == "deterministic":
            continue
        if job.kind not in PIPELINE_STEP_REGISTRY:
            raise OrchidPipelineConfigError(
                f"pipeline {definition.name!r} job {job.id!r} uses kind {job.kind!r}, but no step is "
                "registered for that kind; import its step module before building the pipeline"
            )


# ── Nodes ────────────────────────────────────────────────────


def _make_start_node(definition: CompiledPipeline) -> Callable[..., Any]:
    async def start_node(state: PipelineGraphState, config: RunnableConfig) -> dict[str, Any]:
        session = pipeline_session_from_config(config)
        if auth_from_config(config) is None:
            raise OrchidPipelineAuthError(
                f"pipeline {definition.name!r} requires an authenticated caller; "
                "pass an OrchidAuthContext or user_id/tenant_id"
            )
        session.graph.set_input(dict(session.input), shape_version=session.input_shape_version)
        _seed_objects(session, state.get("seed") or ())
        await session.emit(
            "pipeline.started",
            pipeline=definition.name,
            mode=session.run.mode,
            started_at=session.run.started_at.isoformat(),
        )
        return {"run": session.run.to_dict(), "input": dict(session.input)}

    return start_node


def _seed_objects(session: PipelineRunSession, seeds: Any) -> None:
    if not seeds:
        return
    objects: list[OrchidGraphObject] = []
    for index, seed in enumerate(seeds):
        if not isinstance(seed, Mapping):
            raise OrchidPipelineInputError(f"seed[{index}] must be a mapping")
        object_id = seed.get("id")
        object_type = seed.get("type")
        data = seed.get("data")
        if not isinstance(object_id, str) or not object_id.strip():
            raise OrchidPipelineInputError(f"seed[{index}] requires a non-empty 'id'")
        if not isinstance(object_type, str) or not object_type.strip():
            raise OrchidPipelineInputError(f"seed[{index}] requires a non-empty 'type'")
        if not isinstance(data, Mapping):
            raise OrchidPipelineInputError(f"seed[{index}] requires a 'data' mapping")
        shape_version = str(seed.get("shape_version") or "1")
        validated = dict(data)
        shapes = session.definition.shapes
        if shapes is not None and object_type in shapes:
            try:
                validated = shapes.dump(object_type, data, object_id=object_id, shape_version=shape_version)
            except Exception as exc:
                raise OrchidPipelineInputError(f"seed[{index}] is invalid: {exc}") from exc
        objects.append(
            OrchidGraphObject(
                id=object_id,
                type=object_type,
                data=validated,
                shape_version=shape_version,
                created_by="seed",
                updated_by="seed",
            )
        )
    session.graph.seed(objects)


def _make_join_node() -> Callable[..., Any]:
    async def join_node(state: PipelineGraphState, config: RunnableConfig) -> dict[str, Any]:
        # Convergence point: the router reads the accumulated job results.
        return {}

    return join_node


def _make_join_router(job: CompiledJob) -> Callable[[PipelineGraphState], str]:
    def router(state: PipelineGraphState) -> str:
        results = state.get("job_results") or {}
        # Mutually exclusive branches: only one dependency fires. A skipped
        # active branch still lets its dependents run (spec §7.7.1).
        return "run" if any(dependency in results for dependency in job.depends_on) else "skip"

    return router


def _make_decision_router(job: CompiledJob) -> Callable[[PipelineGraphState], str]:
    labels = set(getattr(job.config, "labels", ()) or ())
    fallback_target: str | None = None
    if job.fallback is not None:
        if job.fallback in job.branches:
            fallback_target = job.branches[job.fallback]
        elif job.fallback not in labels:
            fallback_target = job.fallback

    def router(state: PipelineGraphState) -> str:
        results = state.get("job_results") or {}
        result = results.get(job.id) or {}
        output = result.get("output")
        label = output.get("label") if isinstance(output, Mapping) else None
        target = job.branches.get(label) if isinstance(label, str) else None
        if target is None:
            target = fallback_target
        return target or _FINISH_NODE

    return router


def _make_finish_node() -> Callable[..., Any]:
    async def finish_node(state: PipelineGraphState, config: RunnableConfig) -> dict[str, Any]:
        session = pipeline_session_from_config(config)
        try:
            session.check_deadline()
        except OrchidPipelineTimeoutError:
            # A last job may overrun the run deadline without a following job
            # to observe it; the run is still cancelled, not succeeded.
            session.cancel()
        try:
            session.check_limits()
        except OrchidPipelineError as exc:
            return {
                "status": "failed",
                "output": None,
                "summary": "",
                "error": str(exc),
            }
        return {"status": final_status(session), "output": None, "summary": ""}

    return finish_node


def final_status(session: PipelineRunSession) -> str:
    """Derive the run status from the session (spec §7.12.2).

    A fatal ``on_error: fail`` failure outranks a cancellation: the run failed,
    even if a deadline later cancelled its remaining jobs.
    """
    failed = [result for result in session.job_results.values() if result.status == "failed"]
    for result in failed:
        job = session.definition.jobs.get(result.job_id)
        if job is not None and job.on_error == "fail":
            return "failed"
    if session.cancelled:
        return "cancelled"
    return "partial" if failed else "succeeded"


def _make_job_node(
    definition: CompiledPipeline,
    job: CompiledJob,
    deps: PipelineDependencies | None,
    execution: PipelineExecution,
) -> Callable[..., Any]:
    async def job_node(state: PipelineGraphState, config: RunnableConfig) -> dict[str, Any]:
        session = pipeline_session_from_config(config)
        started = time.perf_counter()

        if session.fatal:
            return await _record_skip(session, job, "a previous job failed", started)
        if session.cancelled:
            return await _record_terminal(session, job, "cancelled", "the run was cancelled", started)
        try:
            session.check_deadline()
        except OrchidPipelineTimeoutError:
            # Caller/deadline cancellation is terminal for the run (§7.12.2).
            session.cancel()
            return await _record_terminal(session, job, "cancelled", "the run deadline was exceeded", started)
        try:
            session.check_limits()
        except OrchidPipelineError as exc:
            return await _record_failure(session, job, exc, started, attempts=1)

        context = session.mapping_context()
        if job.when is not None:
            try:
                gate = evaluate_when(job.when, context, path=f"job {job.id!r} when")
            except OrchidPipelineMappingError as exc:
                return await _record_failure(session, job, exc, started, attempts=1)
            if not gate:
                return await _record_skip(session, job, "when expression evaluated to false", started)

        step = _resolve_step(job, deps, execution)
        attempts = 0
        try:
            step_result, attempts = await _execute_with_retry(
                step,
                job,
                session=session,
                auth=auth_from_config(config),
                execution=execution,
                mapping_context=context,
            )
        except OrchidPipelineError as exc:
            consumed = max(1, attempts, int(getattr(exc, "attempts", 1) or 1))
            return await _record_failure(session, job, exc, started, attempts=consumed)
        except GraphBubbleUp as exc:
            # v1 has no HITL/checkpointer: an interrupt cannot be resumed, and
            # LangGraph returns without a finish status, so record the failure
            # here before letting the control-flow signal through.
            if isinstance(exc, GraphInterrupt):
                interrupted = PipelineJobResult(
                    job_id=job.id,
                    kind=job.kind,
                    status="failed",
                    error="pipeline interrupts are not supported in v1",
                    error_code="pipeline_interrupt_unsupported",
                )
                await session.record_job_result(interrupted)
            raise
        except Exception as exc:  # a step must never crash the graph
            wrapped = OrchidPipelineStepError(
                f"job {job.id!r} failed: {exc}",
                job_id=job.id,
                kind=job.kind,
            )
            return await _record_failure(session, job, wrapped, started, attempts=max(1, attempts))

        try:
            normalized_output = _jsonable(step_result.output)
        except OrchidPipelineSerializationError:
            wrapped = OrchidPipelineStepError(
                f"job {job.id!r} output is not JSON-serializable",
                job_id=job.id,
                kind=job.kind,
            )
            wrapped.usage = dict(step_result.usage)
            return await _record_failure(session, job, wrapped, started, attempts=attempts)
        step_result = OrchidPipelineStepResult(
            output=normalized_output,
            mutations=step_result.mutations,
            usage=step_result.usage,
            metadata=step_result.metadata,
        )

        try:
            mutations, aliases = _build_mutations(job, step_result, context, session)
            extra_objects: tuple[str, ...] = ()
            extra_relations: tuple[str, ...] = ()
            if job.kind == "subpipeline" and bool(getattr(job.config, "merge_graph", False)):
                prefix = f"{job.id}:"
                extra_objects = tuple(
                    str(value)
                    for value in (step_result.metadata.get("_merge_objects") or ())
                    if str(value).startswith(prefix)
                )
                extra_relations = tuple(
                    str(value)
                    for value in (step_result.metadata.get("_merge_relations") or ())
                    if str(value).count(f":{job.id}:") >= 2
                )
            commit = await session.commit(
                job.id,
                mutations,
                aliases=aliases,
                extra_objects=extra_objects,
                extra_relations=extra_relations,
            )
        except OrchidPipelineError as exc:
            exc.usage = dict(step_result.usage)
            return await _record_failure(session, job, exc, started, attempts=attempts)
        except (TypeError, ValueError) as exc:
            wrapped = OrchidPipelineStepError(str(exc), job_id=job.id, kind=job.kind)
            wrapped.usage = dict(step_result.usage)
            return await _record_failure(session, job, wrapped, started, attempts=attempts)

        usage = step_result.usage if isinstance(step_result.usage, Mapping) else {}
        result = PipelineJobResult(
            job_id=job.id,
            kind=job.kind,
            status="succeeded",
            output=step_result.output,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            llm_calls=_usage_int(usage, "llm_calls"),
            prompt_tokens=_usage_int(usage, "prompt_tokens"),
            completion_tokens=_usage_int(usage, "completion_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            objects_written=len(commit.object_ids),
            relations_written=len(commit.relation_ids),
            attempts=attempts,
            metadata={
                **dict(job.metadata),
                **{key: value for key, value in step_result.metadata.items() if not key.startswith("_")},
            },
            objects=dict(commit.aliases),
        )
        await session.record_job_result(result)
        return {"job_results": {job.id: result.to_dict()}}

    return job_node


def _resolve_step(
    job: CompiledJob,
    deps: PipelineDependencies | None,
    execution: PipelineExecution,
) -> OrchidPipelineStep:
    if isinstance(job.handler, OrchidPipelineStep):
        return job.handler
    if job.kind == "deterministic" and callable(job.handler):
        from .steps.deterministic import DeterministicStep

        return DeterministicStep(job, handler=job.handler, logger=execution.logger)
    return resolve_pipeline_step(job.kind, job, deps or PipelineDependencies())


async def _execute_with_retry(
    step: OrchidPipelineStep,
    job: CompiledJob,
    *,
    session: PipelineRunSession,
    auth: Any,
    execution: PipelineExecution,
    mapping_context: Any = None,
    item: Any | None = None,
    index: int | None = None,
    parent_job: str | None = None,
    idempotency_key: str | None = None,
    emit_started: bool = True,
) -> tuple[OrchidPipelineStepResult, int]:
    attempts = 0
    while True:
        attempts += 1
        if emit_started:
            await session.job_started(job.id, attempt=attempts)
        step_context = PipelineStepContext(
            run=session.run,
            job=OrchidPipelineJobContext(
                job_id=job.id,
                kind=job.kind,
                attempt=attempts,
                params=dict(job.params),
                item=item,
                index=index,
                parent_job=parent_job,
            ),
            input=session.input,
            graph=session.graph.view(),
            auth=auth,
            chat_model=execution.chat_model,
            reader=execution.reader,
            logger=execution.logger,
            idempotency_key=idempotency_key or f"{session.run.run_id}:{job.id}:{attempts}",
            tool_caller=execution.tool_caller or call_tool,
            signal_emitter=execution.signal_emitter,
            child_runner=execution.child_runner,
            usage_sink=session.usage_sink,
            llm_calls=session.llm_calls,
            params=dict(job.params),
            pipeline=session.definition,
            mapping_context=mapping_context,
        )
        try:
            return await step.execute(step_context), attempts
        except OrchidPipelineError as exc:
            if attempts > job.retry.max or not _is_retryable(exc):
                # Surface how many attempts the job consumed; the job node
                # reports it without a second source of truth.
                exc.attempts = attempts
                raise
            await asyncio.sleep(_backoff_delay(job, attempts))


def _is_retryable(error: OrchidPipelineError) -> bool:
    return isinstance(error, OrchidPipelineRetryableError) or bool(getattr(error, "retryable", False))


def _backoff_delay(job: CompiledJob, attempt: int) -> float:
    policy = job.retry
    if policy.backoff == "fixed":
        delay = policy.initial_delay_seconds
    elif policy.backoff == "linear":
        delay = policy.initial_delay_seconds * attempt
    else:
        delay = policy.initial_delay_seconds * (2 ** (attempt - 1))
    delay = min(delay, policy.max_delay_seconds, _MAX_BACKOFF_SECONDS)
    if policy.jitter:
        delay *= 0.5 + random.random() / 2
    return delay


# ── Mutations ────────────────────────────────────────────────


def _build_mutations(
    job: CompiledJob,
    step_result: OrchidPipelineStepResult,
    context: Any,
    session: PipelineRunSession,
) -> tuple[list[OrchidGraphMutation], dict[str, str]]:
    mutations: list[OrchidGraphMutation] = list(step_result.mutations)
    aliases: dict[str, str] = {}
    path = f"job {job.id!r}"
    # ``writes.from`` and ``expected_revision`` may read the current step's
    # output as ``result``; gates and the session allowlist use the documented
    # context only, so both layers agree on whether a write is allowed.
    gate_context: Mapping[str, Any] = context.to_dict()
    write_context: Mapping[str, Any] = {**gate_context, "result": step_result.output}
    batch_types: dict[str, str] = {}

    for write in job.writes:
        if write.when is not None and not evaluate_when(write.when, gate_context, path=f"{path} write when"):
            continue
        object_id = render_template(write.id, gate_context, path=f"{path} write id")
        data = (
            evaluate(write.from_, write_context, path=f"{path} write from {object_id!r}")
            if write.from_ is not None
            else step_result.output
        )
        if not isinstance(data, Mapping):
            raise OrchidPipelineStepError(
                f"{path} write {object_id!r} produced {type(data).__name__}, expected a mapping",
                job_id=job.id,
                kind=job.kind,
            )
        expected_revision = write.expected_revision
        if isinstance(expected_revision, str):
            evaluated = evaluate(expected_revision, write_context, path=f"{path} expected_revision")
            if isinstance(evaluated, bool) or not isinstance(evaluated, int) or evaluated < 1:
                raise OrchidPipelineStepError(
                    f"{path} write {object_id!r} expected_revision resolved to {evaluated!r}, expected an integer >= 1",
                    job_id=job.id,
                    kind=job.kind,
                )
            expected_revision = evaluated
        mutations.append(
            OrchidGraphMutation(
                op="upsert" if write.mode == "replace" else write.mode,
                object_id=object_id,
                object_type=write.shape,
                data=dict(data),
                mode="replace" if write.mode == "replace" else "merge",
                expected_revision=expected_revision,
            )
        )
        batch_types[object_id] = write.shape
        if write.as_ is not None:
            aliases[write.as_] = object_id

    for link in job.links:
        if link.when is not None and not evaluate_when(link.when, gate_context, path=f"{path} link when"):
            continue
        source = render_template(link.from_, gate_context, path=f"{path} link from")
        target = render_template(link.to, gate_context, path=f"{path} link to")
        data = render_value(link.data or {}, gate_context, path=f"{path} link data")
        relation = OrchidGraphRelation(
            id=f"{link.type}:{source}:{target}",
            type=link.type,
            source_id=source,
            target_id=target,
            data=data if isinstance(data, Mapping) else {},
        )
        _check_relation_endpoint_types(session, link.type, (source, target), job, batch_types)
        mutations.append(OrchidGraphMutation(op="link", relation=relation))

    return mutations, aliases


def _check_relation_endpoint_types(
    session: PipelineRunSession,
    relation_type: str,
    endpoints: tuple[str, str],
    job: CompiledJob,
    batch_types: Mapping[str, str],
) -> None:
    definition = session.definition
    relation = definition.relations.get(relation_type) if definition.relations else None
    if relation is None:
        return
    sides = ((endpoints[0], relation.from_, "source"), (endpoints[1], relation.to, "target"))
    for object_id, allowed, side in sides:
        obj = session.graph.objects.get(object_id)
        object_type = obj.type if obj is not None else batch_types.get(object_id)
        if object_type is not None and allowed and object_type not in allowed:
            raise OrchidPipelineStepError(
                f"job {job.id!r} link {relation_type!r} {side} {object_id!r} has type {object_type!r}, "
                f"expected one of {', '.join(sorted(allowed))}",
                job_id=job.id,
                kind=job.kind,
            )


# ── Recording helpers ────────────────────────────────────────


async def _record_skip(
    session: PipelineRunSession,
    job: CompiledJob,
    reason: str,
    started: float,
) -> dict[str, Any]:
    result = PipelineJobResult(
        job_id=job.id,
        kind=job.kind,
        status="skipped",
        reason=reason,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    await session.record_job_result(result)
    return {"job_results": {job.id: result.to_dict()}}


async def _record_terminal(
    session: PipelineRunSession,
    job: CompiledJob,
    status: Literal["cancelled"],
    reason: str,
    started: float,
) -> dict[str, Any]:
    result = PipelineJobResult(
        job_id=job.id,
        kind=job.kind,
        status=status,
        reason=reason,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    await session.record_job_result(result)
    return {"job_results": {job.id: result.to_dict()}}


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


async def _record_failure(
    session: PipelineRunSession,
    job: CompiledJob,
    error: OrchidPipelineError,
    started: float,
    *,
    attempts: int,
) -> dict[str, Any]:
    code = getattr(error, "code", "pipeline_step_error")
    usage = getattr(error, "usage", None)
    usage = usage if isinstance(usage, Mapping) else {}
    result = PipelineJobResult(
        job_id=job.id,
        kind=job.kind,
        status="failed",
        error=str(error),
        error_code=code,
        retryable=bool(getattr(error, "retryable", False)),
        attempts=max(1, attempts),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        llm_calls=_usage_int(usage, "llm_calls"),
        prompt_tokens=_usage_int(usage, "prompt_tokens"),
        completion_tokens=_usage_int(usage, "completion_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
    )
    await session.record_job_result(result)
    return {"job_results": {job.id: result.to_dict()}}
