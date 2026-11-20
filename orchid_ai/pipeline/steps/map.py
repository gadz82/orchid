"""Map fan-out topology and helpers for ``kind: map`` jobs.

A map job is a topology kind, not a normal step: the builder wires a coordinator,
one item node and a join. The coordinator evaluates ``over``, applies the map
limits and emits one ``Send`` per item; each item node runs the referenced
template job with ``item``/``index``/``parent_job`` bound and commits that item's
declared writes with the item context; the join aggregates slots
``"<map_job>#<index>"`` and records the single map job result.

The template job never runs as a normal node. Its dependencies are therefore
meaningless and the builder rejects a job that depends on a template.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.types import Send

from ...core.pipeline import (
    OrchidPipelineConfigError,
    OrchidPipelineError,
    OrchidPipelineLimitError,
    OrchidPipelineMappingError,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
    _jsonable,
)
from ...core.run_config import auth_from_config
from ..definition import CompiledJob, CompiledPipeline, PipelineDependencies
from ..mapping import evaluate, evaluate_when, render_template
from ..session import PipelineJobResult, _bounded

__all__ = [
    "MapAggregate",
    "MapItemContext",
    "MapNodes",
    "MapRuntime",
    "aggregate_map_items",
    "build_item_context",
    "build_map_nodes",
    "evaluate_map_items",
    "item_slot",
]

_MAX_ITEM_ERROR_CHARS = 2000


# ORCHID-MEM: 01a1f2c5-f502-76a8-a19c-ed1039340805 — Map item context; the documented run context plus item, index, parent-job and configured binding.
class MapItemContext(dict):
    """JMESPath context for one map item; ``to_dict`` keeps builder helpers happy."""

    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True, slots=True)
class MapAggregate:
    """The map job output and the metrics the join records."""

    output: dict[str, Any]
    failures: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, Any], ...]
    cancelled: tuple[dict[str, Any], ...]
    usage: dict[str, int] = field(default_factory=dict)
    objects_written: int = 0
    relations_written: int = 0


@dataclass(frozen=True, slots=True)
class MapRuntime:
    """Builder-owned job runtime injected into the map nodes."""

    execute_with_retry: Any
    build_mutations: Any
    record_skip: Any
    record_failure: Any
    record_terminal: Any
    resolve_step: Any
    usage_int: Any
    session_from_config: Any


@dataclass(frozen=True, slots=True)
class MapNodes:
    """Callables and node names the builder wires for one map job."""

    coordinator: Any
    router: Any
    item: Any
    join: Any
    item_node: str
    join_node: str


def item_slot(map_id: str, index: int) -> str:
    """The bounded result slot for one item."""
    return f"{map_id}#{index}"


def evaluate_map_items(job: CompiledJob, context: Any, max_items: int) -> list[Any]:
    """Evaluate ``over`` and enforce the list shape and item cap."""
    value = evaluate(
        job.config.over,
        context,
        path=f"job {job.id!r} over",
    )
    if not isinstance(value, list):
        raise OrchidPipelineStepError(
            f"map job {job.id!r} 'over' must resolve to a list, got {type(value).__name__}",
            job_id=job.id,
            kind="map",
        )
    if len(value) > max_items:
        raise OrchidPipelineLimitError(
            f"map job {job.id!r} resolved {len(value)} items, exceeding max_map_items={max_items}"
        )
    return list(value)


def build_item_context(
    base: Any,
    binding: str,
    item: Any,
    index: int,
    parent_job: str,
) -> MapItemContext:
    """Add the configured item binding and the map metadata to a render context."""
    if hasattr(base, "to_dict"):
        values = dict(base.to_dict())
    else:
        values = dict(base)
    values[binding] = item
    values["item"] = item
    values["index"] = index
    values["parent_job"] = parent_job
    return MapItemContext(values)


def aggregate_map_items(records: list[dict[str, Any]]) -> MapAggregate:
    """Aggregate item records into the map output and totals.

    ``items`` contains only successful outputs keyed by slot; failures and
    skipped items stay in their own lists so the caller can apply the map policy
    without losing the reason.
    """
    ordered = sorted(records, key=lambda record: int(record.get("index", 0)))
    items: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    cancelled: list[dict[str, Any]] = []
    usage = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    objects_written = 0
    relations_written = 0
    for record in ordered:
        status = record.get("status")
        slot = str(record.get("slot") or "")
        if status == "succeeded":
            items[slot] = record.get("output")
        elif status == "skipped":
            skipped.append(
                {
                    "slot": slot,
                    "index": record.get("index"),
                    "reason": record.get("reason") or "item skipped",
                }
            )
        elif status == "cancelled":
            cancelled.append(
                {
                    "slot": slot,
                    "index": record.get("index"),
                    "reason": record.get("reason") or "item cancelled",
                }
            )
        else:
            failures.append(
                {
                    "slot": slot,
                    "index": record.get("index"),
                    "error": record.get("error") or "map item failed",
                    "error_code": record.get("error_code") or "pipeline_step_error",
                    "retryable": bool(record.get("retryable", False)),
                }
            )
        for key in usage:
            usage[key] += int(record.get(key) or 0)
        objects_written += int(record.get("objects_written") or 0)
        relations_written += int(record.get("relations_written") or 0)
    return MapAggregate(
        output={"items": items, "failures": failures, "skipped": skipped, "cancelled": cancelled},
        failures=tuple(failures),
        skipped=tuple(skipped),
        cancelled=tuple(cancelled),
        usage=usage,
        objects_written=objects_written,
        relations_written=relations_written,
    )


# ORCHID-MEM: 01a1f2c5-f537-7d84-a084-fb6f67ce1ae7 — Map builder; coordinator, item and join nodes with bounded concurrency and item-specific commits.
def build_map_nodes(
    definition: CompiledPipeline,
    job: CompiledJob,
    deps: PipelineDependencies | None,
    execution: Any,
    runtime: MapRuntime,
) -> MapNodes:
    """Build the coordinator/item/join callables for one map job."""
    template = definition.jobs.get(job.map_template or "")
    if template is None:
        raise OrchidPipelineConfigError(f"map job {job.id!r} references unknown template job {job.map_template!r}")
    if template.kind == "map":
        raise OrchidPipelineConfigError(f"map job {job.id!r} maps over another map job; nested maps are not supported")
    if template.kind == "decision" and template.branches:
        raise OrchidPipelineConfigError(f"map job {job.id!r} maps over a branching decision job")
    if template.object_aliases:
        raise OrchidPipelineConfigError(
            f"map job {job.id!r} template {template.id!r} declares aliases; "
            "map item writes must use item-specific ids instead"
        )
    template_step = runtime.resolve_step(template, deps, execution)
    map_config = job.config
    item_node = f"{job.id}__item"
    join_node = f"{job.id}__join"

    async def coordinator(state: Any, config: RunnableConfig) -> dict[str, Any]:
        session = runtime.session_from_config(config)
        started = time.perf_counter()
        if session.fatal:
            recorded = await runtime.record_skip(session, job, "a previous job failed", started)
            return {**recorded, "map_plan": {job.id: {"status": "skipped"}}}
        if session.cancelled:
            recorded = await runtime.record_terminal(session, job, "cancelled", "the run was cancelled", started)
            return {**recorded, "map_plan": {job.id: {"status": "cancelled"}}}
        try:
            session.check_deadline()
        except OrchidPipelineTimeoutError:
            session.cancel()
            recorded = await runtime.record_terminal(
                session, job, "cancelled", "the run deadline was exceeded", started
            )
            return {**recorded, "map_plan": {job.id: {"status": "cancelled"}}}
        try:
            session.check_limits()
        except OrchidPipelineError as exc:
            recorded = await runtime.record_failure(session, job, exc, started, attempts=1)
            return {**recorded, "map_plan": {job.id: {"status": "failed"}}}

        context = session.mapping_context()
        if job.when is not None:
            try:
                gate = evaluate_when(job.when, context, path=f"job {job.id!r} when")
            except OrchidPipelineMappingError as exc:
                recorded = await runtime.record_failure(session, job, exc, started, attempts=1)
                return {**recorded, "map_plan": {job.id: {"status": "failed"}}}
            if not gate:
                recorded = await runtime.record_skip(session, job, "when expression evaluated to false", started)
                return {**recorded, "map_plan": {job.id: {"status": "skipped"}}}

        try:
            items = evaluate_map_items(job, context, session.limits.max_map_items)
        except OrchidPipelineError as exc:
            recorded = await runtime.record_failure(session, job, exc, started, attempts=1)
            return {**recorded, "map_plan": {job.id: {"status": "failed"}}}

        try:
            _validate_item_write_targets(
                template,
                job,
                items,
                context,
                map_config.as_,
                job.id,
                session.limits.max_prompt_chars,
            )
        except OrchidPipelineError as exc:
            recorded = await runtime.record_failure(session, job, exc, started, attempts=1)
            return {**recorded, "map_plan": {job.id: {"status": "failed"}}}
        except Exception as exc:
            wrapped = OrchidPipelineStepError(
                f"map job {job.id!r} write-target validation failed ({type(exc).__name__})",
                job_id=job.id,
                kind="map",
            )
            recorded = await runtime.record_failure(session, job, wrapped, started, attempts=1)
            return {**recorded, "map_plan": {job.id: {"status": "failed"}}}

        concurrency = min(int(map_config.concurrency), int(session.limits.max_parallel))
        session.map_semaphore(job.id, concurrency)
        await session.job_started(job.id)
        deadline_at = time.monotonic() + float(job.timeout_seconds) if job.timeout_seconds is not None else None
        header = {
            "status": "run",
            "template": template.id,
            "binding": map_config.as_,
            "concurrency": concurrency,
            "started_at": started,
            "deadline_at": deadline_at,
        }
        return {
            "map_plan": {job.id: {**header, "items": items}},
            "map_header": {job.id: header},
        }

    def router(state: Any) -> Any:
        plan = (state.get("map_plan") or {}).get(job.id) or {}
        if plan.get("status") != "run" or not plan.get("items"):
            return join_node
        header = {key: value for key, value in plan.items() if key != "items"}
        return [
            Send(
                item_node,
                {
                    "_map_item": {
                        "map_id": job.id,
                        "template": template.id,
                        "binding": map_config.as_,
                        "concurrency": plan["concurrency"],
                        "index": index,
                        "item": item,
                        "header": header,
                    }
                },
            )
            for index, item in enumerate(plan["items"])
        ]

    async def item_node_fn(state: Any, config: RunnableConfig) -> dict[str, Any]:
        session = runtime.session_from_config(config)
        spec = state.get("_map_item") or {}
        header = spec.get("header") or {}
        map_id = str(spec.get("map_id") or job.id)
        index = int(spec.get("index") or 0)
        item = spec.get("item")
        slot = item_slot(map_id, index)
        started = time.perf_counter()
        header_update = {"map_header": {map_id: header}}

        def cancelled(reason: str) -> dict[str, Any]:
            return {
                "map_items": {slot: _cancelled_record(slot, index, reason, started)},
                **header_update,
            }

        if session.cancelled:
            return cancelled("the run was cancelled")
        try:
            session.check_deadline()
        except OrchidPipelineTimeoutError:
            session.cancel()
            return cancelled("the run deadline was exceeded")

        context = build_item_context(
            session.mapping_context(),
            str(spec.get("binding") or map_config.as_),
            item,
            index,
            map_id,
        )

        if template.when is not None:
            try:
                gate = evaluate_when(template.when, context, path=f"job {template.id!r} when")
            except OrchidPipelineMappingError as exc:
                return {
                    "map_items": {slot: _failed_record(slot, index, exc, started, attempts=1)},
                    **header_update,
                }
            if not gate:
                return {
                    "map_items": {
                        slot: {
                            "slot": slot,
                            "index": index,
                            "status": "skipped",
                            "reason": "when expression evaluated to false",
                            "duration_ms": (time.perf_counter() - started) * 1000.0,
                        }
                    },
                    **header_update,
                }

        async with session.map_semaphore(map_id, int(spec.get("concurrency") or 1)):
            if session.cancelled:
                return cancelled("the run was cancelled")
            try:
                session.check_deadline()
            except OrchidPipelineTimeoutError:
                session.cancel()
                return cancelled("the run deadline was exceeded")

            attempts = 0
            timeout: float | None = None
            deadline_at = header.get("deadline_at")
            if deadline_at is not None:
                timeout = float(deadline_at) - time.monotonic()
                if timeout <= 0:
                    return {
                        "map_items": {
                            slot: _failed_record(
                                slot,
                                index,
                                OrchidPipelineTimeoutError(f"map job {map_id!r} timed out before item {index} started"),
                                started,
                                attempts=1,
                            )
                        },
                        **header_update,
                    }
            try:
                call = runtime.execute_with_retry(
                    template_step,
                    template,
                    session=session,
                    auth=auth_from_config(config),
                    execution=execution,
                    mapping_context=context,
                    item=item,
                    index=index,
                    parent_job=map_id,
                    idempotency_key=f"{session.run.run_id}:{map_id}:{index}",
                    emit_started=False,
                )
                step_result, attempts = (
                    await asyncio.wait_for(call, timeout=timeout) if timeout is not None else await call
                )
                if session.cancelled:
                    return cancelled("the run was cancelled")
                normalized = _jsonable(step_result.output)
                step_result = OrchidPipelineStepResult(
                    output=normalized,
                    mutations=step_result.mutations,
                    usage=step_result.usage,
                    metadata=step_result.metadata,
                )
                mutations, aliases = runtime.build_mutations(template, step_result, context, session)
                commit = await session.commit(template.id, mutations, aliases=aliases, context=context)
                usage = step_result.usage if isinstance(step_result.usage, Mapping) else {}
                record = {
                    "slot": slot,
                    "index": index,
                    "status": "succeeded",
                    "output": normalized,
                    "duration_ms": (time.perf_counter() - started) * 1000.0,
                    "llm_calls": runtime.usage_int(usage, "llm_calls"),
                    "prompt_tokens": runtime.usage_int(usage, "prompt_tokens"),
                    "completion_tokens": runtime.usage_int(usage, "completion_tokens"),
                    "total_tokens": runtime.usage_int(usage, "total_tokens"),
                    "objects_written": len(commit.object_ids),
                    "relations_written": len(commit.relation_ids),
                    "attempts": attempts,
                    "metadata": dict(step_result.metadata),
                }
            except GraphBubbleUp:
                raise
            except TimeoutError:
                record = _failed_record(
                    slot,
                    index,
                    OrchidPipelineTimeoutError(f"map item {slot!r} timed out"),
                    started,
                    attempts=max(1, attempts),
                )
            except OrchidPipelineError as exc:
                record = _failed_record(slot, index, exc, started, attempts=max(1, attempts))
            except Exception as exc:
                record = _failed_record(
                    slot,
                    index,
                    OrchidPipelineStepError(
                        f"map item {slot!r} failed ({type(exc).__name__})",
                        job_id=map_id,
                        kind="map",
                    ),
                    started,
                    attempts=max(1, attempts),
                )
        return {"map_items": {slot: record}, **header_update}

    async def join_node_fn(state: Any, config: RunnableConfig) -> dict[str, Any]:
        session = runtime.session_from_config(config)
        header = (state.get("map_header") or {}).get(job.id) or (state.get("map_plan") or {}).get(job.id) or {}
        if header.get("status") != "run":
            return {}
        records = [record for slot, record in (state.get("map_items") or {}).items() if slot.startswith(f"{job.id}#")]
        aggregate = aggregate_map_items(records)
        started = float(header.get("started_at") or time.perf_counter())
        usage = aggregate.usage
        try:
            session.check_deadline()
        except OrchidPipelineTimeoutError:
            session.cancel()

        if session.cancelled:
            cancelled_result = PipelineJobResult(
                job_id=job.id,
                kind="map",
                status="cancelled",
                output=aggregate.output,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                llm_calls=runtime.usage_int(usage, "llm_calls"),
                prompt_tokens=runtime.usage_int(usage, "prompt_tokens"),
                completion_tokens=runtime.usage_int(usage, "completion_tokens"),
                total_tokens=runtime.usage_int(usage, "total_tokens"),
                objects_written=aggregate.objects_written,
                relations_written=aggregate.relations_written,
                attempts=1,
                metadata={
                    **dict(job.metadata),
                    "item_failures": len(aggregate.failures),
                    "item_skips": len(aggregate.skipped),
                    "item_cancelled": len(aggregate.cancelled),
                },
            )
            await session.record_job_result(cancelled_result)
            return {"job_results": {job.id: cancelled_result.to_dict()}}

        if aggregate.failures and map_config.on_item_error == "fail":
            failed = PipelineJobResult(
                job_id=job.id,
                kind="map",
                status="failed",
                output=aggregate.output,
                error=f"map job {job.id!r} failed for {len(aggregate.failures)} item(s)",
                error_code="pipeline_step_error",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                llm_calls=runtime.usage_int(usage, "llm_calls"),
                prompt_tokens=runtime.usage_int(usage, "prompt_tokens"),
                completion_tokens=runtime.usage_int(usage, "completion_tokens"),
                total_tokens=runtime.usage_int(usage, "total_tokens"),
                objects_written=aggregate.objects_written,
                relations_written=aggregate.relations_written,
                attempts=1,
                metadata={
                    **dict(job.metadata),
                    "item_failures": len(aggregate.failures),
                    "item_skips": len(aggregate.skipped),
                    "item_cancelled": len(aggregate.cancelled),
                },
            )
            await session.record_job_result(failed)
            return {"job_results": {job.id: failed.to_dict()}}

        step_result = OrchidPipelineStepResult(
            output=aggregate.output,
            usage=aggregate.usage,
            metadata={
                "item_failures": len(aggregate.failures),
                "item_skips": len(aggregate.skipped),
                "item_cancelled": len(aggregate.cancelled),
            },
        )
        try:
            mutations, aliases = runtime.build_mutations(job, step_result, session.mapping_context(), session)
            commit = await session.commit(job.id, mutations, aliases=aliases)
        except OrchidPipelineError as exc:
            return await runtime.record_failure(session, job, exc, started, attempts=1)
        except (TypeError, ValueError) as exc:
            wrapped = OrchidPipelineStepError(str(exc), job_id=job.id, kind="map")
            return await runtime.record_failure(session, job, wrapped, started, attempts=1)

        result = PipelineJobResult(
            job_id=job.id,
            kind="map",
            status="succeeded",
            output=aggregate.output,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            llm_calls=runtime.usage_int(usage, "llm_calls"),
            prompt_tokens=runtime.usage_int(usage, "prompt_tokens"),
            completion_tokens=runtime.usage_int(usage, "completion_tokens"),
            total_tokens=runtime.usage_int(usage, "total_tokens"),
            objects_written=aggregate.objects_written + len(commit.object_ids),
            relations_written=aggregate.relations_written + len(commit.relation_ids),
            attempts=1,
            metadata={
                **dict(job.metadata),
                "item_failures": len(aggregate.failures),
                "item_skips": len(aggregate.skipped),
                "item_cancelled": len(aggregate.cancelled),
            },
            objects=dict(commit.aliases),
        )
        await session.record_job_result(result)
        return {"job_results": {job.id: result.to_dict()}}

    return MapNodes(
        coordinator=coordinator,
        router=router,
        item=item_node_fn,
        join=join_node_fn,
        item_node=item_node,
        join_node=join_node,
    )


def _render_write_targets(
    job: CompiledJob,
    context: Any,
    max_chars: int,
) -> tuple[list[str], list[str]]:
    objects: list[str] = []
    relations: list[str] = []
    for write in job.writes:
        if write.when is not None and not evaluate_when(write.when, context, path=f"job {job.id!r} write when"):
            continue
        objects.append(render_template(write.id, context, path=f"job {job.id!r} write id", max_chars=max_chars))
    for link in job.links:
        if link.when is not None and not evaluate_when(link.when, context, path=f"job {job.id!r} link when"):
            continue
        source = render_template(link.from_, context, path=f"job {job.id!r} link from", max_chars=max_chars)
        target = render_template(link.to, context, path=f"job {job.id!r} link to", max_chars=max_chars)
        relations.append(f"{link.type}:{source}:{target}")
    return objects, relations


def _validate_item_write_targets(
    template: CompiledJob,
    map_job: CompiledJob,
    items: list[Any],
    base_context: Any,
    binding: str,
    map_id: str,
    max_chars: int,
) -> None:
    """Reject template/map writes whose rendered targets are not item-specific."""
    seen_objects: dict[str, int | str] = {}
    seen_relations: dict[str, int | str] = {}
    map_objects, map_relations = _render_write_targets(map_job, base_context, max_chars)
    for target in map_objects:
        seen_objects[target] = "map"
    for target in map_relations:
        seen_relations[target] = "map"

    for index, item in enumerate(items):
        context = build_item_context(base_context, binding, item, index, map_id)
        if template.when is not None and not evaluate_when(template.when, context, path=f"job {template.id!r} when"):
            continue
        objects, relations = _render_write_targets(template, context, max_chars)
        for kind, targets in (("write", objects), ("relation", relations)):
            seen_item: set[str] = set()
            for target in targets:
                if target in seen_item:
                    raise OrchidPipelineStepError(
                        f"map job {map_id!r} {kind} target {target!r} is declared twice in item {index}",
                        job_id=map_id,
                        kind="map",
                    )
                seen_item.add(target)
        for target in objects:
            if target in seen_objects:
                raise OrchidPipelineStepError(
                    f"map job {map_id!r} write target {target!r} is not item-specific "
                    f"(already used by {seen_objects[target]!r} before item {index})",
                    job_id=map_id,
                    kind="map",
                )
            seen_objects[target] = index
        for target in relations:
            if target in seen_relations:
                raise OrchidPipelineStepError(
                    f"map job {map_id!r} relation target {target!r} is not item-specific "
                    f"(already used by {seen_relations[target]!r} before item {index})",
                    job_id=map_id,
                    kind="map",
                )
            seen_relations[target] = index


def _cancelled_record(slot: str, index: int, reason: str, started: float) -> dict[str, Any]:
    return {
        "slot": slot,
        "index": index,
        "status": "cancelled",
        "reason": reason[:_MAX_ITEM_ERROR_CHARS],
        "duration_ms": (time.perf_counter() - started) * 1000.0,
    }


def _failed_record(
    slot: str,
    index: int,
    error: OrchidPipelineError,
    started: float,
    *,
    attempts: int,
) -> dict[str, Any]:
    message = _bounded(str(error), _MAX_ITEM_ERROR_CHARS) or "map item failed"
    return {
        "slot": slot,
        "index": index,
        "status": "failed",
        "error": message,
        "error_code": getattr(error, "code", "pipeline_step_error"),
        "retryable": bool(getattr(error, "retryable", False)),
        "duration_ms": (time.perf_counter() - started) * 1000.0,
        "attempts": attempts,
    }
