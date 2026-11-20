"""Pipeline definition compiler: validated YAML to executable, read-only plans.

Configuration validation (:mod:`orchid_ai.config.schema_pipeline` and
:mod:`orchid_ai.config.schema_agent`) owns the structural rule matrix: names,
references, DAG cycles, branch/join composition, map/sub-pipeline rules, limits
and template *syntax*. This module owns what can only happen at build time:

* compile shapes and share one :class:`OrchidShapeRegistry`;
* compute the effective limits (a definition may tighten, never raise, a cap);
* resolve dotted ``uses`` (a callable) and ``class`` (an ``OrchidPipelineStep``)
  paths once, failing the build with a configuration error;
* check the reference set the compiler and runner depend on (dependency ids,
  branch targets, map templates, sub-pipeline targets) and the shape-field
  references a field-based shape only knows at compile time (input defaults and
  output mapping keys);
* keep a slot for the compiled LangGraph graph, filled by the builder.

Configuration validation owns the structural rule matrix for YAML/DB loads. The
compiler re-checks the subset a programmatic definition can bypass (cycles,
joins, sub-pipeline depth, parallel static writes, write/link references), so
both entry points produce an executable plan or a precise error.

Nothing here imports a consumer class at configuration-parse time: resolution
happens when :func:`compile_pipeline` runs, which is the build boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config.pipeline_templates import validate_expression, validate_template
from ..config.schema_pipeline import (
    OrchidPipelineAgentJobConfig,
    OrchidPipelineCustomJobConfig,
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDefinitionLimitsConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineEmitJobConfig,
    OrchidPipelineJobConfig,
    OrchidPipelineLimitsConfig,
    OrchidPipelineLLMJobConfig,
    OrchidPipelineMapJobConfig,
    OrchidPipelineObjectWriteConfig,
    OrchidPipelineRelationWriteConfig,
    OrchidPipelineRetryConfig,
    OrchidPipelinesConfig,
    OrchidPipelineSubPipelineJobConfig,
)
from ..core.pipeline import OrchidPipelineConfigError, OrchidPipelineMappingError, OrchidPipelineStep
from ..utils import import_class
from .mapping import references_context_key
from .schema import OrchidCompiledShape, OrchidShapeRegistry, ShapeCompiler

__all__ = [
    "CompiledJob",
    "CompiledPipeline",
    "PipelineDependencies",
    "PipelineLimits",
    "compile_pipeline",
    "compile_pipelines",
]

_LIMIT_FIELDS = (
    "timeout_seconds",
    "max_jobs",
    "max_objects",
    "max_relations",
    "max_object_bytes",
    "max_graph_bytes",
    "max_llm_calls",
    "max_prompt_chars",
    "max_context_chars",
    "max_map_items",
    "max_parallel",
    "max_subpipeline_depth",
)


# ── Limits ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PipelineLimits:
    """Effective hard caps for one run.

    ``from_config`` takes the tighter of the global cap and the definition
    override. Configuration validation already rejects an override above the
    global cap; the ``min`` here keeps the invariant true for definitions built
    programmatically.
    """

    timeout_seconds: float | None = None
    max_jobs: int = 100
    max_objects: int = 10_000
    max_relations: int = 50_000
    max_object_bytes: int = 1_000_000
    max_graph_bytes: int = 20_000_000
    max_llm_calls: int = 50
    max_prompt_chars: int = 32_000
    max_context_chars: int = 8_000
    max_map_items: int = 100
    max_parallel: int = 8
    max_subpipeline_depth: int = 3

    @classmethod
    def from_config(
        cls,
        limits: OrchidPipelineLimitsConfig | None = None,
        overrides: OrchidPipelineDefinitionLimitsConfig | None = None,
    ) -> PipelineLimits:
        global_limits = limits or OrchidPipelineLimitsConfig()
        definition_limits = overrides or OrchidPipelineDefinitionLimitsConfig()
        values: dict[str, Any] = {}
        for field_name in _LIMIT_FIELDS:
            values[field_name] = _tighter(getattr(global_limits, field_name), getattr(definition_limits, field_name))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in _LIMIT_FIELDS}


# ── Dependencies ─────────────────────────────────────────────


# ORCHID-MEM: 01a1ef10-9e6b-7940-9e70-abd464b553f1 — Pipeline dependencies also carry agent configs and graph-built agent executors for guardrail/routing parity.
@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Runtime objects a compiled definition may reference.

    ``None`` means "not checked": a programmatic compile that has no graph yet
    compiles name references without resolving them. The lifecycle (P-007)
    passes the real maps so a missing agent, decision model or tool fails the
    build instead of the first run.

    ``agent_configs`` maps a top-level leaf agent name to its configuration
    (mini-agent options). ``agent_executors`` maps the same name to the
    graph-built :class:`~orchid_ai.graph.agent_executor.OrchidAgentExecutor`,
    which carries the per-agent and global guardrail chains and the model
    router. ``pipeline_resolver`` returns a compiled child pipeline for a
    sub-pipeline name once the registry is built. New fields are appended so
    positional construction keeps its historical meaning.
    """

    agents: Mapping[str, Any] | None = None
    decision_models: Mapping[str, Any] | None = None
    tools: Mapping[str, Any] | None = None
    signal_emitter: Any | None = None
    agent_configs: Mapping[str, Any] | None = None
    agent_executors: Mapping[str, Any] | None = None
    pipeline_resolver: Callable[[str], Any] | None = None


# ── Compiled artifacts ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CompiledJob:
    """One job in executable form.

    ``config`` is the typed Pydantic job model; the flat fields below are the
    values the session and runner need on every execution. ``handler`` is set
    only for kinds the compiler resolves itself: a deterministic callable
    (``uses``) or a shared custom-step instance (``class``). Other kinds are
    executed by a registered step factory (see
    :mod:`orchid_ai.pipeline.registry`).
    """

    id: str
    kind: str
    config: OrchidPipelineJobConfig
    depends_on: tuple[str, ...] = ()
    join: str = "all"
    when: str | None = None
    writes: tuple[OrchidPipelineObjectWriteConfig, ...] = ()
    links: tuple[OrchidPipelineRelationWriteConfig, ...] = ()
    retry: OrchidPipelineRetryConfig = field(default_factory=OrchidPipelineRetryConfig)
    timeout_seconds: float | None = None
    on_error: str = "fail"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    handler: Any = None
    map_template: str | None = None
    subpipeline: str | None = None
    branches: Mapping[str, str] = field(default_factory=dict)
    fallback: str | None = None
    allow_dynamic_writes: bool = True

    @property
    def params(self) -> Mapping[str, Any]:
        """Step parameters from the typed config (empty for kinds without them)."""
        params = getattr(self.config, "params", None)
        return params if isinstance(params, Mapping) else {}

    @property
    def object_aliases(self) -> tuple[str, ...]:
        """Aliases this job may register, in declaration order."""
        return tuple(write.as_ for write in self.writes if write.as_ is not None)


# ORCHID-MEM: 01a1eb1b-d932-7fbb-b67d-58a9bddd96d8 — Compiled pipeline; validated config to plans, effective limits, resolved imports and the graph cache slot.
@dataclass(slots=True)
class CompiledPipeline:
    """A definition ready to build and run.

    ``graph`` is the cache slot the builder fills (P-007); it is ``None`` until
    then. Every other field is set at compilation and treated as read-only.
    """

    name: str
    version: str
    description: str
    definition: OrchidPipelineDefinitionConfig
    jobs: Mapping[str, CompiledJob]
    limits: PipelineLimits
    shapes: OrchidShapeRegistry | None = None
    relations: Mapping[str, Any] = field(default_factory=dict)
    input_shape: OrchidCompiledShape | None = None
    output_shape: OrchidCompiledShape | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    _graph: Any = field(default=None, repr=False)

    @property
    def graph(self) -> Any:
        """The compiled LangGraph graph, or ``None`` before the builder runs."""
        return self._graph

    def set_graph(self, graph: Any) -> None:
        """Cache the compiled graph; the builder calls this once per build."""
        if self._graph is not None:
            raise OrchidPipelineConfigError(f"pipeline {self.name!r}: compiled graph is already set")
        self._graph = graph

    @property
    def enabled(self) -> bool:
        return self.definition.enabled

    @property
    def exposed(self) -> bool:
        return self.definition.expose.as_tool

    @property
    def ordered_jobs(self) -> tuple[CompiledJob, ...]:
        """Jobs in declaration order; the builder derives the topology itself."""
        return tuple(self.jobs.values())


# ── Compilation ──────────────────────────────────────────────


def compile_pipelines(
    pipelines: OrchidPipelinesConfig | None,
    *,
    deps: PipelineDependencies | None = None,
    resolve_import: Callable[[str], Any] | None = None,
) -> dict[str, CompiledPipeline]:
    """Compile every enabled definition of a ``pipelines:`` block.

    Shapes are compiled once for the whole block. A disabled block compiles to
    an empty mapping, and a disabled definition is skipped.
    """
    if pipelines is None or not pipelines.enabled:
        return {}
    resolve = resolve_import or import_class
    compiler = _build_compiler(pipelines, resolve_class=resolve)
    shapes = compiler.compile()
    compiled: dict[str, CompiledPipeline] = {}
    for name, definition in pipelines.definitions.items():
        if not definition.enabled:
            continue
        compiled[name] = _compile_one(
            name,
            definition,
            pipelines=pipelines,
            compiler=compiler,
            shapes=shapes,
            deps=deps,
            resolve_import=resolve,
        )
    return compiled


def compile_pipeline(
    name: str,
    definition: OrchidPipelineDefinitionConfig,
    *,
    pipelines: OrchidPipelinesConfig | None = None,
    deps: PipelineDependencies | None = None,
    resolve_import: Callable[[str], Any] | None = None,
    shape_compiler: ShapeCompiler | None = None,
) -> CompiledPipeline:
    """Compile one definition.

    ``pipelines`` supplies the global limits, defaults, relations and shapes;
    omit it for a definition compiled programmatically (no declared shapes,
    default limits). ``shape_compiler`` lets a caller share one compiler across
    definitions instead of recompiling the block per definition.
    """
    if not isinstance(name, str) or not name.strip():
        raise OrchidPipelineConfigError("pipeline name must be a non-empty string")
    resolve = resolve_import or import_class
    compiler = shape_compiler or _build_compiler(pipelines, resolve_class=resolve)
    return _compile_one(
        name,
        definition,
        pipelines=pipelines,
        compiler=compiler,
        shapes=compiler.compile(),
        deps=deps,
        resolve_import=resolve,
    )


def _tighter(ceiling: Any, override: Any) -> Any:
    """Return the tighter of a global cap and a definition override."""
    if ceiling is None:
        return override
    if override is None:
        return ceiling
    return min(ceiling, override)


def _build_compiler(
    pipelines: OrchidPipelinesConfig | None,
    *,
    resolve_class: Callable[[str], Any],
) -> ShapeCompiler:
    block = pipelines or OrchidPipelinesConfig()
    return ShapeCompiler(
        block.shapes,
        resolve_class=resolve_class,
        allow_shape_version_mismatch=block.allow_shape_version_mismatch,
    )


def _compile_one(
    name: str,
    definition: OrchidPipelineDefinitionConfig,
    *,
    pipelines: OrchidPipelinesConfig | None,
    compiler: ShapeCompiler,
    shapes: OrchidShapeRegistry | None,
    deps: PipelineDependencies | None,
    resolve_import: Callable[[str], Any],
) -> CompiledPipeline:
    errors: list[str] = []
    block = pipelines or OrchidPipelinesConfig()
    defaults = block.defaults
    limits = PipelineLimits.from_config(block.limits, definition.limits)
    if len(definition.jobs) > limits.max_jobs:  # defensive; config rejects it
        errors.append(f"declares {len(definition.jobs)} jobs, exceeding limits.max_jobs={limits.max_jobs}")

    input_shape = _compile_input_shape(name, definition, compiler, shapes, errors)
    output_shape = (
        _require_shape(shapes, definition.output.shape, label=f"pipeline {name!r} output", errors=errors)
        if definition.output is not None and definition.output.shape is not None
        else None
    )
    errors.extend(_validate_shape_fields(name, definition, input_shape, output_shape))
    errors.extend(_validate_structure(name, definition, block, shapes, limits))

    compiled_jobs: dict[str, CompiledJob] = {}
    for job_id, job in definition.jobs.items():
        label = f"pipeline {name!r} job {job_id!r}"
        errors.extend(_validate_templates(label, job))
        errors.extend(_validate_job_refs(name, job_id, job, definition, block, deps))
        compiled_jobs[job_id] = _compile_job(
            job_id,
            job,
            defaults=defaults,
            limits=limits,
            shape_registry=shapes,
            resolve_import=resolve_import,
            errors=errors,
        )

    if errors:
        raise OrchidPipelineConfigError("invalid pipeline definition: " + "; ".join(errors))

    return CompiledPipeline(
        name=name,
        version=definition.version,
        description=definition.description,
        definition=definition,
        jobs=compiled_jobs,
        limits=limits,
        shapes=shapes,
        relations=block.relations,
        input_shape=input_shape,
        output_shape=output_shape,
        defaults={
            "model": defaults.model,
            "temperature": defaults.temperature,
            "max_tokens": defaults.max_tokens,
            "retry": defaults.retry,
            "timeout_seconds": defaults.timeout_seconds,
        },
    )


def _compile_job(
    job_id: str,
    job: OrchidPipelineJobConfig,
    *,
    defaults: Any,
    limits: PipelineLimits,
    shape_registry: OrchidShapeRegistry | None,
    resolve_import: Callable[[str], Any],
    errors: list[str],
) -> CompiledJob:
    handler: Any = None
    map_template: str | None = None
    subpipeline: str | None = None
    branches: Mapping[str, str] = {}
    fallback: str | None = None
    allow_dynamic_writes = True

    if isinstance(job, OrchidPipelineDeterministicJobConfig):
        handler = _resolve_import(job.uses, label=f"job {job_id!r} 'uses'", resolve=resolve_import, errors=errors)
        allow_dynamic_writes = job.allow_dynamic_writes
    elif isinstance(job, OrchidPipelineCustomJobConfig):
        handler = _build_custom_step(job.class_path, job_id=job_id, resolve=resolve_import, errors=errors)
    elif isinstance(job, OrchidPipelineMapJobConfig):
        map_template = job.job
    elif isinstance(job, OrchidPipelineSubPipelineJobConfig):
        subpipeline = job.pipeline
    elif isinstance(job, OrchidPipelineDecisionJobConfig):
        branches = dict(job.branches)
        fallback = job.fallback
    elif isinstance(job, (OrchidPipelineLLMJobConfig, OrchidPipelineAgentJobConfig)):
        output = job.output
        if output is not None and output.shape is not None:
            _require_shape(shape_registry, output.shape, label=f"job {job_id!r} output", errors=errors)
    elif isinstance(job, OrchidPipelineEmitJobConfig):
        pass

    retry = job.retry or defaults.retry
    timeout_seconds = job.timeout_seconds if job.timeout_seconds is not None else defaults.timeout_seconds
    if timeout_seconds is not None and limits.timeout_seconds is not None:
        # Limits are hard caps: an inherited default or a per-job value cannot
        # raise the run timeout above the effective cap.
        timeout_seconds = min(timeout_seconds, limits.timeout_seconds)
    return CompiledJob(
        id=job_id,
        kind=job.kind,
        config=job,
        depends_on=tuple(job.depends_on),
        join=job.join,
        when=job.when,
        writes=tuple(job.writes),
        links=tuple(job.links),
        retry=retry,
        timeout_seconds=timeout_seconds,
        on_error=job.on_error,
        metadata=dict(job.metadata),
        handler=handler,
        map_template=map_template,
        subpipeline=subpipeline,
        branches=branches,
        fallback=fallback,
        allow_dynamic_writes=allow_dynamic_writes,
    )


# ── Build-time validation ────────────────────────────────────


def _compile_input_shape(
    pipeline_name: str,
    definition: OrchidPipelineDefinitionConfig,
    compiler: ShapeCompiler,
    shapes: OrchidShapeRegistry | None,
    errors: list[str],
) -> OrchidCompiledShape | None:
    """Compile a named input shape or an inline ``input.fields`` declaration."""
    input_config = definition.input
    if input_config is None:
        return None
    if input_config.shape is not None:
        return _require_shape(shapes, input_config.shape, label=f"pipeline {pipeline_name!r} input", errors=errors)
    try:
        return compiler.compile_inline(
            f"{pipeline_name}_input",
            input_config.fields or {},
            description=f"Input of pipeline {pipeline_name}",
            version=definition.version,
        )
    except Exception as exc:
        errors.append(f"pipeline {pipeline_name!r} input fields cannot be compiled: {exc}")
        return None


def _validate_structure(
    pipeline_name: str,
    definition: OrchidPipelineDefinitionConfig,
    block: OrchidPipelinesConfig,
    shapes: OrchidShapeRegistry | None,
    limits: PipelineLimits,
) -> list[str]:
    """Re-check the structural rules a programmatic definition can bypass.

    Configuration validation owns this matrix for YAML/DB loads. The compiler
    repeats the rules that make a definition executable (cycles, joins,
    sub-pipeline depth, parallel static writes, write/link references), so a
    definition assembled in code cannot produce a broken run either.
    """
    errors: list[str] = []
    label = f"pipeline {pipeline_name!r}"
    jobs = definition.jobs

    cycle = _find_cycle(jobs)
    if cycle:
        errors.append(f"{label} dependency graph contains a cycle: " + " -> ".join(cycle))

    for job_id, job in jobs.items():
        if job.join == "any" and len(job.depends_on) < 2:
            errors.append(f"{label} job {job_id!r} uses join: any but declares fewer than two dependencies")
        for write in job.writes:
            if shapes is not None and write.shape not in shapes:
                errors.append(f"{label} job {job_id!r} writes to unknown shape {write.shape!r}")
        for link in job.links:
            if not block.allow_untyped_relations and link.type not in block.relations:
                errors.append(f"{label} job {job_id!r} links unknown relation {link.type!r}")

    errors.extend(_validate_subpipeline_depth(pipeline_name, definition, block, limits.max_subpipeline_depth))
    errors.extend(_validate_branch_topology(pipeline_name, jobs, _branch_membership(jobs, include_fallback=False)))
    errors.extend(_validate_conditional_target_dependencies(pipeline_name, jobs))

    membership = _branch_membership(jobs, include_fallback=True)
    aliases: dict[str, list[str]] = {}
    for job_id, job in jobs.items():
        for write in job.writes:
            if write.as_ is not None:
                aliases.setdefault(write.as_, []).append(job_id)
    for alias, owners in aliases.items():
        for index, first in enumerate(owners):
            for second in owners[index + 1 :]:
                if not _mutually_exclusive(first, second, membership):
                    errors.append(
                        f"{label} alias {alias!r} is declared by jobs {first!r} and {second!r} "
                        "that are not mutually exclusive"
                    )

    if not block.allow_parallel_writes:
        static_writes: dict[str, list[str]] = {}
        for job_id, job in jobs.items():
            for object_id in _static_write_ids(job):
                static_writes.setdefault(object_id, []).append(job_id)
        for object_id, owners in static_writes.items():
            for index, first in enumerate(owners):
                for second in owners[index + 1 :]:
                    if _depends_on(first, second, jobs) or _depends_on(second, first, jobs):
                        continue
                    if _mutually_exclusive(first, second, membership):
                        continue
                    errors.append(
                        f"{label} parallel jobs {first!r} and {second!r} both write static object {object_id!r}"
                    )
    return errors


def _validate_branch_topology(
    pipeline_name: str,
    jobs: Mapping[str, OrchidPipelineJobConfig],
    membership: Mapping[str, list[tuple[str, str]]],
) -> list[str]:
    """Reject a branch target that lists its decision or feeds a ``join: all``."""
    errors: list[str] = []
    label = f"pipeline {pipeline_name!r}"

    for target, owners in membership.items():
        target_job = jobs.get(target)
        if target_job is None:
            continue
        for decision_id, _branch_label in owners:
            if decision_id in target_job.depends_on:
                errors.append(
                    f"{label} branch target {target!r} must not list decision job {decision_id!r} in depends_on"
                )

    for job_id, job in jobs.items():
        if job.join != "all":
            continue
        owners_by_decision: dict[str, set[str]] = {}
        for dependency in job.depends_on:
            for decision_id, branch_label in membership.get(dependency, []):
                owners_by_decision.setdefault(decision_id, set()).add(branch_label)
        for decision_id, branch_labels in owners_by_decision.items():
            if len(branch_labels) > 1:
                errors.append(
                    f"{label} job {job_id!r} uses join: all over mutually exclusive branches "
                    f"of decision {decision_id!r}; use join: any"
                )

    for target in membership:
        if target not in jobs:
            continue
        for dependent_id, dependent in jobs.items():
            if target in dependent.depends_on and dependent.join != "any":
                errors.append(f"{label} branch target {target!r} feeds job {dependent_id!r} which must use join: any")
    return errors


def _validate_conditional_target_dependencies(
    pipeline_name: str,
    jobs: Mapping[str, OrchidPipelineJobConfig],
) -> list[str]:
    """A branch target or job-id fallback may depend only on its decision (§7.9.2)."""
    errors: list[str] = []
    label = f"pipeline {pipeline_name!r}"
    owners: dict[str, set[str]] = {}
    for job_id, job in jobs.items():
        if not isinstance(job, OrchidPipelineDecisionJobConfig):
            continue
        targets = list(job.branches.values())
        if (
            job.fallback is not None
            and job.fallback not in job.branches
            and job.fallback not in set(job.labels)
            and job.fallback in jobs
        ):
            targets.append(job.fallback)
        for target in targets:
            owners.setdefault(target, set()).add(job_id)
    for target, decisions in owners.items():
        target_job = jobs.get(target)
        if target_job is None:
            continue
        unexpected = [dependency for dependency in target_job.depends_on if dependency not in decisions]
        if unexpected:
            errors.append(
                f"{label} job {target!r} is a decision target and must depend only on its decision job; "
                f"unexpected dependencies: {', '.join(sorted(unexpected))}"
            )
    return errors


def _branch_membership(
    jobs: Mapping[str, OrchidPipelineJobConfig],
    *,
    include_fallback: bool,
) -> dict[str, list[tuple[str, str]]]:
    """Map a target to the ``(decision job, label)`` pairs that select it.

    ``include_fallback`` is for the mutual-exclusion checks only: a job-id
    fallback is selected when no branch matches, so it cannot run alongside a
    branch target of the same decision. It is *not* part of branch topology —
    fallback jobs need not obey the branch-target *join* rules (spec §7.6.4
    rule 9). They still obey the decision-target dependency rule: a fallback may
    depend on its decision (or on nothing), never on an unrelated job.
    """
    membership: dict[str, list[tuple[str, str]]] = {}
    for job_id, job in jobs.items():
        if not isinstance(job, OrchidPipelineDecisionJobConfig):
            continue
        for label, target in job.branches.items():
            membership.setdefault(target, []).append((job_id, label))
        if include_fallback and job.fallback is not None and job.fallback not in job.labels:
            membership.setdefault(job.fallback, []).append((job_id, "__fallback__"))
    return membership


def _mutually_exclusive(
    left: str,
    right: str,
    membership: Mapping[str, list[tuple[str, str]]],
) -> bool:
    for decision_id, label in membership.get(left, []):
        for other_decision, other_label in membership.get(right, []):
            if decision_id == other_decision and label != other_label:
                return True
    return False


def _static_write_ids(job: OrchidPipelineJobConfig) -> set[str]:
    return {write.id for write in job.writes if "{{" not in write.id}


def _depends_on(job_id: str, target: str, jobs: Mapping[str, OrchidPipelineJobConfig]) -> bool:
    stack = [job_id]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        job = jobs.get(current)
        if job is None:
            continue
        if target in job.depends_on:
            return True
        stack.extend(job.depends_on)
    return False


def _find_cycle(jobs: Mapping[str, OrchidPipelineJobConfig]) -> list[str]:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dependency in jobs[node].depends_on:
            if dependency not in jobs:
                continue
            marker = state.get(dependency, 0)
            if marker == 1:
                return stack[stack.index(dependency) :] + [dependency]
            if marker == 0:
                found = visit(dependency)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for job_id in jobs:
        if state.get(job_id, 0) == 0:
            found = visit(job_id)
            if found:
                return found
    return []


def _validate_subpipeline_depth(
    pipeline_name: str,
    definition: OrchidPipelineDefinitionConfig,
    block: OrchidPipelinesConfig,
    max_depth: int,
) -> list[str]:
    """Check the sub-pipeline chain is acyclic and within the 1-based depth cap."""
    errors: list[str] = []
    stack: list[tuple[OrchidPipelineDefinitionConfig, int, tuple[str, ...]]] = [(definition, 1, (pipeline_name,))]
    while stack:
        current, depth, path = stack.pop()
        if depth > max_depth:
            errors.append(f"pipeline {pipeline_name!r} exceeds max_subpipeline_depth={max_depth}: " + " -> ".join(path))
            return errors
        for job in current.jobs.values():
            if not isinstance(job, OrchidPipelineSubPipelineJobConfig):
                continue
            child = block.definitions.get(job.pipeline)
            if child is None:
                continue
            if job.pipeline in path:
                errors.append(
                    f"pipeline {pipeline_name!r} sub-pipeline cycle detected: " + " -> ".join((*path, job.pipeline))
                )
                return errors
            stack.append((child, depth + 1, (*path, job.pipeline)))
    return errors


def _validate_job_refs(
    pipeline_name: str,
    job_id: str,
    job: OrchidPipelineJobConfig,
    definition: OrchidPipelineDefinitionConfig,
    block: OrchidPipelinesConfig,
    deps: PipelineDependencies | None,
) -> list[str]:
    label = f"pipeline {pipeline_name!r} job {job_id!r}"
    errors: list[str] = []
    jobs = definition.jobs

    for dependency in job.depends_on:
        if dependency == job_id:
            errors.append(f"{label} must not depend on itself")
        elif dependency not in jobs:
            errors.append(f"{label} depends on unknown job {dependency!r}")

    if isinstance(job, OrchidPipelineMapJobConfig):
        template = jobs.get(job.job)
        if template is None:
            errors.append(f"{label} maps over unknown template job {job.job!r}")
        elif template.kind == "map":
            errors.append(f"{label} maps over job {job.job!r}, which is itself a map job")
        elif template.kind == "decision" and getattr(template, "branches", None):
            errors.append(f"{label} maps over job {job.job!r}, which is a branching decision job")

    if isinstance(job, OrchidPipelineDecisionJobConfig):
        for branch, target in job.branches.items():
            if target != job.fallback and target not in jobs:
                errors.append(f"{label} branch {branch!r} targets unknown job {target!r}")
        if job.fallback is not None and job.fallback not in jobs and job.fallback not in job.labels:
            errors.append(f"{label} fallback {job.fallback!r} is neither a label nor a job id")

    if isinstance(job, OrchidPipelineSubPipelineJobConfig):
        target = block.definitions.get(job.pipeline)
        if target is None:
            errors.append(f"{label} references unknown sub-pipeline {job.pipeline!r}")
        elif target.enabled is False:
            errors.append(f"{label} references disabled sub-pipeline {job.pipeline!r}")

    if deps is not None:
        if isinstance(job, OrchidPipelineAgentJobConfig) and deps.agents is not None and job.agent not in deps.agents:
            errors.append(f"{label} references unknown agent {job.agent!r}")
        if (
            isinstance(job, OrchidPipelineDecisionJobConfig)
            and deps.decision_models is not None
            and job.decision_model not in deps.decision_models
        ):
            errors.append(f"{label} references unknown decision model {job.decision_model!r}")
        if isinstance(job, OrchidPipelineLLMJobConfig) and deps.tools is not None:
            for tool_name in job.tools:
                if tool_name not in deps.tools:
                    errors.append(f"{label} references unknown tool {tool_name!r}")

    return errors


def _validate_templates(label: str, job: OrchidPipelineJobConfig) -> list[str]:
    """Re-validate every declaration template and expression at build time.

    Configuration validation already does this; repeating it here means a
    definition built programmatically (tests, adapters) cannot reach a render
    with an expression the renderer or configuration validator would reject.
    The shared grammar is the only rule set, so this cannot drift.
    """
    errors: list[str] = []
    templates: list[tuple[str, str | None]] = []
    expressions: list[tuple[str, str | None]] = []
    gates: list[tuple[str, str | None]] = [("when", job.when)]

    if isinstance(job, OrchidPipelineLLMJobConfig):
        templates.extend([("prompt", job.prompt), ("system_prompt", job.system_prompt)])
    elif isinstance(job, OrchidPipelineAgentJobConfig):
        templates.append(("prompt", job.prompt))
        if job.input is not None:
            templates.extend(_walk_strings(job.input, path="input"))
    elif isinstance(job, OrchidPipelineDecisionJobConfig):
        templates.append(("question", job.question))
        expressions.append(("context_from", job.context_from))
    elif isinstance(job, OrchidPipelineMapJobConfig):
        expressions.append(("over", job.over))
    elif isinstance(job, OrchidPipelineEmitJobConfig):
        templates.extend(_walk_strings(job.payload, path="payload"))
        templates.append(("dedupe_key", job.dedupe_key))
    elif isinstance(job, OrchidPipelineSubPipelineJobConfig) and job.input:
        templates.extend(_walk_strings(job.input, path="input"))

    for write in job.writes:
        templates.append(("write.id", write.id))
        expressions.append(("write.from", write.from_))
        if isinstance(write.expected_revision, str) and not write.expected_revision.isdigit():
            expressions.append(("write.expected_revision", write.expected_revision))
        gates.append(("write.when", write.when))
    for link in job.links:
        templates.extend([("link.from", link.from_), ("link.to", link.to)])
        gates.append(("link.when", link.when))
        if link.data:
            templates.extend(_walk_strings(link.data, path="link.data"))

    for field_name, value in templates:
        if value is None:
            continue
        try:
            validate_template(value, path=f"{label} {field_name}")
        except OrchidPipelineMappingError as exc:
            errors.append(str(exc))
    for field_name, value in expressions:
        if value is None:
            continue
        try:
            validate_expression(value, path=f"{label} {field_name}")
        except OrchidPipelineMappingError as exc:
            errors.append(str(exc))
    for field_name, value in gates:
        if value is None:
            continue
        try:
            if "{{" in value or "}}" in value:
                validate_template(value, path=f"{label} {field_name}")
            else:
                validate_expression(value, path=f"{label} {field_name}")
        except OrchidPipelineMappingError as exc:
            errors.append(str(exc))

    return errors


def _walk_strings(value: Any, *, path: str) -> list[tuple[str, str | None]]:
    """Flatten a nested mapping/list into ``(path, string)`` pairs."""
    found: list[tuple[str, str | None]] = []
    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_walk_strings(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, path=f"{path}[{index}]"))
    return found


# ORCHID-MEM: 01a1ef68-3f52-7e20-80bc-a45ee6934c30 — Output mapping summary is available only for none/caller interpretation.
def _validate_shape_fields(
    pipeline_name: str,
    definition: OrchidPipelineDefinitionConfig,
    input_shape: OrchidCompiledShape | None,
    output_shape: OrchidCompiledShape | None,
) -> list[str]:
    """Check defaults and output mapping keys against compiled shape fields.

    Only possible after compilation: a class-based shape's fields are unknown at
    parse time.
    """
    errors: list[str] = []
    label = f"pipeline {pipeline_name!r}"

    if definition.input is not None and input_shape is not None:
        declared = _shape_field_names(input_shape)
        if declared is not None:
            for key in definition.input.defaults:
                if key not in declared:
                    errors.append(f"{label} input defaults reference unknown field {key!r}")

    if definition.output is not None and output_shape is not None:
        if definition.output.mapping is not None:
            declared = _shape_field_names(output_shape)
            if declared is not None:
                for key in definition.output.mapping:
                    if key not in declared:
                        errors.append(f"{label} output mapping references unknown field {key!r}")
        if definition.interpretation.mode in ("template", "llm"):
            for key, expression in (definition.output.mapping or {}).items():
                if references_context_key(expression, "summary"):
                    errors.append(
                        f"{label} output mapping field {key!r} references 'summary', which is "
                        f"unavailable when interpretation.mode is {definition.interpretation.mode!r}"
                    )
            if definition.output.from_ is not None and references_context_key(definition.output.from_, "summary"):
                errors.append(
                    f"{label} output 'from' references 'summary', which is unavailable when "
                    f"interpretation.mode is {definition.interpretation.mode!r}"
                )

    return errors


def _shape_field_names(shape: OrchidCompiledShape) -> frozenset[str] | None:
    model = shape.model
    model_fields = getattr(model, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return frozenset(str(name) for name in model_fields)
    dataclass_fields = getattr(model, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, Mapping):
        return frozenset(str(name) for name in dataclass_fields if not str(name).startswith("_"))
    return None


def _require_shape(
    shapes: OrchidShapeRegistry | None,
    name: str,
    *,
    label: str,
    errors: list[str],
) -> OrchidCompiledShape | None:
    if shapes is None:
        return None
    try:
        return shapes.get(name)
    except Exception as exc:  # unknown shape is a build error, not a crash
        errors.append(f"{label} references unknown shape {name!r}: {exc}")
        return None


def _resolve_import(
    path: str,
    *,
    label: str,
    resolve: Callable[[str], Any],
    errors: list[str],
) -> Any:
    try:
        handler = resolve(path)
    except Exception as exc:
        errors.append(f"{label} cannot import {path!r}: {exc}")
        return None
    if not callable(handler):
        errors.append(f"{label} {path!r} is not callable")
        return None
    return handler


def _build_custom_step(
    class_path: str,
    *,
    job_id: str,
    resolve: Callable[[str], Any],
    errors: list[str],
) -> Any:
    """Import and instantiate a custom step once per custom job."""
    label = f"job {job_id!r} 'class'"
    step_class = _resolve_import(class_path, label=label, resolve=resolve, errors=errors)
    if step_class is None:
        return None
    if not isinstance(step_class, type) or not issubclass(step_class, OrchidPipelineStep):
        errors.append(f"{label} {class_path!r} must subclass OrchidPipelineStep")
        return None
    try:
        return step_class()
    except Exception as exc:
        errors.append(f"{label} {class_path!r} cannot be instantiated without arguments: {exc}")
        return None
