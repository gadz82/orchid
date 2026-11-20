"""Sub-pipeline step: one child pipeline with inherited identity and graph merge.

A ``kind: subpipeline`` job renders its input, asks the parent runner's
``run_pipeline`` collaborator to execute the child on the same runner instance,
selects the configured ``output_from`` path and, only with
``merge_graph: true``, prefixes every child object/relation with
``"<job_id>:"`` before returning the mutations. The child runs with the parent
auth, mode, chat scope, correlation id and remaining deadline; the child's LLM
usage is returned to the parent job.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from ...config.schema_pipeline import OrchidPipelineSubPipelineJobConfig
from ...core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    OrchidGraphMutation,
    OrchidGraphRelation,
    OrchidPipelineConfigError,
    OrchidPipelineError,
    OrchidPipelineGraphConflictError,
    OrchidPipelineLimitError,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from ..definition import CompiledJob, CompiledPipeline
from ..mapping import build_mapping_context, evaluate, render_value

__all__ = ["SubPipelineStep"]

_MAX_ERROR_CHARS = 500


# ORCHID-MEM: 01a1f2fe-ccb4-765f-93b0-74c18717ddbd — Sub-pipeline step; child runner reuse with inherited auth/depth/deadline and prefixed graph merge.
class SubPipelineStep(OrchidPipelineStep):
    """Execute a ``kind: subpipeline`` job."""

    kind = "subpipeline"

    def __init__(
        self,
        job: CompiledJob,
        *,
        resolver: Callable[[str], CompiledPipeline | None] | None = None,
        logger: Any | None = None,
    ) -> None:
        if not isinstance(job, CompiledJob):
            raise TypeError("SubPipelineStep requires a CompiledJob")
        if not isinstance(job.config, OrchidPipelineSubPipelineJobConfig):
            raise TypeError("SubPipelineStep requires an OrchidPipelineSubPipelineJobConfig")
        self._job = job
        self._config = job.config
        self._resolver = resolver
        self._logger = logger

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        """Run the child and return its selected output plus explicit mutations."""
        child = self._resolve_child()
        depth = int(getattr(ctx.run, "subpipeline_depth", 0)) + 1
        max_depth = self._max_depth(ctx)
        if depth >= max_depth:
            raise OrchidPipelineLimitError(
                f"sub-pipeline job {self._job.id!r} depth {depth} exceeds max_subpipeline_depth={max_depth}"
            )
        child_runner = getattr(ctx, "child_runner", None)
        if child_runner is None:
            raise OrchidPipelineConfigError(
                f"sub-pipeline job {self._job.id!r} has no child runner; build the pipeline through"
                " OrchidPipelineRunner"
            )

        context = getattr(ctx, "mapping_context", None)
        if context is None:
            context = build_mapping_context(input=ctx.input, run=ctx.run)
        rendered_input = render_value(
            self._config.input or {},
            context,
            path=f"job {self._job.id!r} input",
            strict=True,
        )
        if not isinstance(rendered_input, Mapping):
            raise OrchidPipelineStepError(
                f"sub-pipeline job {self._job.id!r} input must render to an object",
                job_id=self._job.id,
                kind=self.kind,
            )

        mode = "chat" if ctx.run.mode == "chat" else "direct"
        chat_id = ctx.run.chat_id if mode == "chat" else None
        timeout = self._remaining_timeout(ctx)
        usage_out: dict[str, int] = {}
        try:
            call = child_runner(
                child,
                dict(rendered_input),
                auth=ctx.auth,
                mode=mode,
                chat_id=chat_id,
                correlation_id=ctx.run.correlation_id,
                parent_run_id=ctx.run.run_id,
                subpipeline_depth=depth,
                subpipeline_max_depth=self._max_depth(ctx),
                timeout=timeout,
                include_graph=bool(self._config.merge_graph),
                usage_out=usage_out,
            )
            result = await asyncio.wait_for(call, timeout=timeout) if timeout is not None else await call
        except TimeoutError as exc:
            suffix = f" after {timeout:.3f} seconds" if timeout is not None else ""
            error = OrchidPipelineTimeoutError(f"sub-pipeline job {self._job.id!r} timed out{suffix}")
            error.usage = _usage_from_sink(usage_out)
            raise error from exc
        except OrchidPipelineError:
            raise
        except Exception as exc:
            raise OrchidPipelineStepError(
                f"sub-pipeline job {self._job.id!r} failed to run child {child.name!r} ({type(exc).__name__})",
                job_id=self._job.id,
                kind=self.kind,
            ) from exc

        usage = _usage_from_result(result)
        if result.status != "succeeded":
            detail = (result.error or "").strip()[:_MAX_ERROR_CHARS]
            suffix = f": {detail}" if detail else ""
            error = OrchidPipelineStepError(
                f"sub-pipeline job {self._job.id!r} child {child.name!r} finished with "
                f"status {result.status!r}{suffix}",
                job_id=self._job.id,
                kind=self.kind,
            )
            error.usage = usage
            raise error

        try:
            output = self._select_output(result, context)
        except OrchidPipelineError as exc:
            exc.usage = usage
            raise
        self._check_parent_llm_budget(ctx, usage)
        try:
            mutations = self._merge_mutations(ctx, result) if self._config.merge_graph else ()
        except OrchidPipelineError as exc:
            exc.usage = usage
            raise
        return OrchidPipelineStepResult(
            output=output,
            mutations=mutations,
            usage=usage,
            metadata={
                "subpipeline": child.name,
                "child_run_id": result.run_id,
                "child_status": result.status,
                "_merge_objects": tuple(m.object_id for m in mutations if m.object_id),
                "_merge_relations": tuple(m.relation.id for m in mutations if m.relation is not None),
            },
        )

    def _check_parent_llm_budget(self, ctx: Any, usage: Mapping[str, Any]) -> None:
        pipeline = getattr(ctx, "pipeline", None)
        limits = getattr(pipeline, "limits", None)
        max_calls = getattr(limits, "max_llm_calls", None)
        calls = int(usage.get("llm_calls") or 0)
        if max_calls is None or calls == 0:
            return
        used = int(getattr(ctx, "llm_calls", 0)) + calls
        if used > int(max_calls):
            error = OrchidPipelineLimitError(
                f"sub-pipeline job {self._job.id!r} used {used} LLM calls, exceeding max_llm_calls={max_calls}"
            )
            error.usage = dict(usage)
            raise error

    # ── Resolution / bounds ─────────────────────────────────

    def _resolve_child(self) -> CompiledPipeline:
        if self._resolver is None:
            raise OrchidPipelineConfigError(
                f"sub-pipeline job {self._job.id!r} has no pipeline resolver; build through the lifecycle"
            )
        child = self._resolver(self._config.pipeline)
        if child is None:
            raise OrchidPipelineConfigError(
                f"sub-pipeline job {self._job.id!r} references unknown pipeline {self._config.pipeline!r}"
            )
        if not isinstance(child, CompiledPipeline):
            raise OrchidPipelineConfigError(
                f"sub-pipeline job {self._job.id!r} resolver returned {type(child).__name__}, expected CompiledPipeline"
            )
        return child

    def _max_depth(self, ctx: Any) -> int:
        run_depth = getattr(ctx.run, "subpipeline_max_depth", None)
        if run_depth is not None:
            return int(run_depth)
        pipeline = getattr(ctx, "pipeline", None)
        limits = getattr(pipeline, "limits", None)
        max_depth = getattr(limits, "max_subpipeline_depth", None)
        return int(max_depth) if max_depth is not None else 3

    def _remaining_timeout(self, ctx: Any) -> float | None:
        timeout = self._job.timeout_seconds
        deadline = getattr(ctx.run, "deadline", None)
        if deadline is None:
            return timeout
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise OrchidPipelineTimeoutError(
                f"sub-pipeline job {self._job.id!r} cannot start: the run deadline has passed"
            )
        return remaining if timeout is None else min(timeout, remaining)

    # ── Output / graph merge ─────────────────────────────────

    def _select_output(self, result: Any, context: Any) -> Any:
        base = context.to_dict() if hasattr(context, "to_dict") else dict(context)
        child = result.to_dict()
        selection_context = {
            **base,
            "result": child,
            "output": child.get("output"),
            "jobs": child.get("jobs", {}),
            "graph_stats": child.get("graph_stats", {}),
        }
        return evaluate(
            self._config.output_from,
            selection_context,
            path=f"job {self._job.id!r} output_from",
        )

    def _merge_mutations(self, ctx: Any, result: Any) -> tuple[OrchidGraphMutation, ...]:
        if not result.graph:
            if result.metadata.get("graph_omitted") == "size":
                raise OrchidPipelineStepError(
                    f"sub-pipeline job {self._job.id!r} cannot merge the child graph: it was omitted "
                    "by the child graph size cap",
                    job_id=self._job.id,
                    kind=self.kind,
                )
            return ()
        prefix = f"{self._job.id}:"
        existing_objects = set()
        existing_relations = {relation.id for relation in ctx.graph.relations()}
        mutations: list[OrchidGraphMutation] = []
        for child_id, obj in (result.graph.get("objects") or {}).items():
            if child_id == PIPELINE_INPUT_OBJECT_ID or obj.get("type") == PIPELINE_INPUT_OBJECT_TYPE:
                continue
            object_id = prefix + str(child_id)
            if ctx.graph.get(object_id) is not None or object_id in existing_objects:
                raise OrchidPipelineGraphConflictError(
                    f"sub-pipeline job {self._job.id!r} merge collision on object {object_id!r}"
                )
            existing_objects.add(object_id)
            mutations.append(
                OrchidGraphMutation(
                    op="create",
                    object_id=object_id,
                    object_type=str(obj.get("type") or ""),
                    data=dict(obj.get("data") or {}),
                )
            )
        for child_id, rel in (result.graph.get("relations") or {}).items():
            source = str(rel.get("source_id") or "")
            target = str(rel.get("target_id") or "")
            if PIPELINE_INPUT_OBJECT_ID in (source, target):
                continue
            prefixed_source = prefix + source
            prefixed_target = prefix + target
            relation_id = f"{rel.get('type')}:{prefixed_source}:{prefixed_target}"
            if relation_id in existing_relations:
                raise OrchidPipelineGraphConflictError(
                    f"sub-pipeline job {self._job.id!r} merge collision on relation {relation_id!r}"
                )
            existing_relations.add(relation_id)
            mutations.append(
                OrchidGraphMutation(
                    op="link",
                    relation=OrchidGraphRelation(
                        id=relation_id,
                        type=str(rel.get("type") or ""),
                        source_id=prefixed_source,
                        target_id=prefixed_target,
                        data=dict(rel.get("data") or {}),
                        metadata=dict(rel.get("metadata") or {}),
                    ),
                )
            )
        return tuple(mutations)


def _usage_from_result(result: Any) -> dict[str, int]:
    return {
        "llm_calls": int(result.graph_stats.get("llm_calls") or 0),
        "prompt_tokens": int(result.graph_stats.get("prompt_tokens") or 0),
        "completion_tokens": int(result.graph_stats.get("completion_tokens") or 0),
        "total_tokens": int(result.graph_stats.get("total_tokens") or 0),
    }


def _usage_from_sink(sink: Mapping[str, Any]) -> dict[str, int]:
    return {
        "llm_calls": int(sink.get("llm_calls") or 0),
        "prompt_tokens": int(sink.get("prompt_tokens") or 0),
        "completion_tokens": int(sink.get("completion_tokens") or 0),
        "total_tokens": int(sink.get("total_tokens") or 0),
    }
