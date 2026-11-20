"""Run session: the authoritative state of one pipeline run.

A run session owns the information graph and everything the job nodes must not
put in graph state: the input, the per-job results, errors, effective limits,
the observer, the commit lock, the deadline, cancellation and the counters. It
is created by the runner, travels in ``RunnableConfig`` (never serialised) and
is discarded when the run ends.

Two invariants shape this module:

* **Commits are atomic and allowlisted.** A job never mutates the graph directly;
  :meth:`PipelineRunSession.commit` re-renders the job's declared writes against
  the current context, rejects anything outside them with
  :class:`OrchidPipelineWriteNotAllowedError`, and applies the batch
  all-or-nothing under one lock.
* **Observability never breaks a run.** Observer failures and log formatting are
  contained here; a broken observer cannot change the result.

This module also holds the concrete :class:`PipelineStepContext`, the object a
step receives. The port is a ``Protocol`` in ``core/``; the concrete class lives
here because it needs the session's run identity, graph view and collaborators.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal

from ..core.pipeline import (
    OrchidPipelineConfigError,
    OrchidPipelineEvent,
    OrchidPipelineEventsDisabledError,
    OrchidPipelineJobContext,
    OrchidPipelineLimitError,
    OrchidPipelineObserver,
    OrchidPipelineRunContext,
    OrchidPipelineTimeoutError,
    OrchidPipelineWriteNotAllowedError,
)
from .definition import CompiledJob, CompiledPipeline
from .graph import DataGraph, DataGraphCommit
from .mapping import OrchidPipelineMappingContext, build_mapping_context, evaluate_when, render_template

__all__ = ["PipelineJobResult", "PipelineRunSession", "PipelineStepContext"]

_MAX_ERRORS = 50
_MAX_WRITTEN_EVENTS = 100
_MAX_LOG_ERROR_CHARS = 500

JobStatus = Literal["succeeded", "failed", "skipped", "cancelled"]


@dataclass(frozen=True, slots=True)
class PipelineJobResult:
    """What one job produced, in the form the result and the context expose.

    ``objects`` is the alias map the job registered (alias → object id), so a
    downstream template can read ``jobs.<id>.objects.<alias>``.
    """

    job_id: str
    kind: str
    status: JobStatus
    output: Any = None
    error: str | None = None
    error_code: str = ""
    retryable: bool = False
    attempts: int = 1
    duration_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    objects_written: int = 0
    relations_written: int = 0
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    objects: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed", "skipped", "cancelled"}:
            raise ValueError(f"job status must be succeeded, failed, skipped or cancelled, got {self.status!r}")
        for field_name in ("job_id", "kind", "error_code", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"job result {field_name} must be a string")
        for field_name in ("error",):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"job result {field_name} must be a string or None")
        for field_name in (
            "attempts",
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "objects_written",
            "relations_written",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"job result {field_name} must be an integer")
            if value < 0:
                raise ValueError(f"job result {field_name} must be non-negative")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise TypeError("job result duration_ms must be a number")
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("job result duration_ms must be finite and non-negative")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "objects", dict(self.objects))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe job entry used by the result and the context."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "status": self.status,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 3),
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "objects_written": self.objects_written,
            "relations_written": self.relations_written,
            "attempts": self.attempts,
            "objects": dict(self.objects),
            "metadata": dict(self.metadata),
        }
        if self.error is not None:
            payload["error"] = self.error
            payload["error_code"] = self.error_code
            payload["retryable"] = self.retryable
        if self.reason:
            payload["reason"] = self.reason
        return payload


# ORCHID-MEM: 01a1eb1b-d965-71b3-978a-bf493551d3e2 — Run session; authoritative graph, allowlisted atomic commits, limits, deadline, cancellation and lifecycle events.
class PipelineRunSession:
    """Owns one run's graph, results, limits, cancellation and observer.

    ``clock`` is injectable so deadline tests do not sleep; production uses
    :func:`datetime.now`. The input is deep-copied and read-only at the top
    level, and the deadline falls back to the run-start time plus the effective
    ``timeout_seconds`` when the caller did not supply one.
    """

    def __init__(
        self,
        *,
        definition: CompiledPipeline,
        run: OrchidPipelineRunContext,
        input: Mapping[str, Any],
        observer: OrchidPipelineObserver | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], datetime] | None = None,
        usage_out: dict[str, int] | None = None,
    ) -> None:
        if not isinstance(definition, CompiledPipeline):
            raise TypeError("definition must be a CompiledPipeline")
        if not isinstance(run, OrchidPipelineRunContext):
            raise TypeError("run must be an OrchidPipelineRunContext")
        if not isinstance(input, Mapping):
            raise TypeError("input must be a mapping")
        self.definition = definition
        self.run = run
        # Deep-copied and read-only at the top level: a step cannot tamper with
        # the caller's input or replace a top-level key. Nested values are
        # copies, so mutating one does not affect the caller's mapping.
        self.input = MappingProxyType(copy.deepcopy(dict(input)))
        self.limits = definition.limits
        self.started_at = run.started_at
        self.deadline = run.deadline or (
            run.started_at + timedelta(seconds=self.limits.timeout_seconds)
            if self.limits.timeout_seconds is not None
            else None
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._logger = logger or logging.getLogger(__name__)
        self._observer = observer
        self._usage_out = usage_out
        self._job_results: dict[str, PipelineJobResult] = {}
        self._errors: list[dict[str, Any]] = []
        self._llm_calls = 0
        self._objects_written = 0
        self._relations_written = 0
        self._fatal = False
        self._cancel_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._map_semaphores: dict[str, asyncio.Semaphore] = {}
        self._shapes_validator = definition.shapes.dump if definition.shapes is not None else None
        self.graph = DataGraph(
            validator=self._shapes_validator,
            max_objects=self.limits.max_objects,
            max_relations=self.limits.max_relations,
            max_object_bytes=self.limits.max_object_bytes,
            max_graph_bytes=self.limits.max_graph_bytes,
            shape_versions=(
                {shape_name: definition.shapes.get(shape_name).version for shape_name in definition.shapes}
                if definition.shapes is not None
                else None
            ),
        )

    # ── Read-only state ──────────────────────────────────────

    @property
    def job_results(self) -> Mapping[str, PipelineJobResult]:
        return dict(self._job_results)

    @property
    def errors(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._errors)

    @property
    def fatal(self) -> bool:
        """True once a job failed with the ``fail`` error policy."""
        return self._fatal

    @property
    def cancelled(self) -> bool:
        """True once the caller cancelled the run."""
        return self._cancel_event.is_set()

    @property
    def llm_calls(self) -> int:
        return max(self._llm_calls, int((self._usage_out or {}).get("llm_calls", 0)))

    @property
    def usage_sink(self) -> dict[str, int] | None:
        """Mutable usage sink shared with a nested child run, or ``None``."""
        return self._usage_out

    # ORCHID-MEM: 01a1ef3d-7a1c-7b70-ac04-cbcb6ad798bb — Session LLM counter; interpretation calls count toward the run cap.
    def record_llm_call(self, count: int = 1) -> None:
        """Count LLM calls made outside a job (for example interpretation).

        Job results already carry their own usage and are added by
        :meth:`record_job_result`. Interpretation runs after the graph, so it
        needs the same run-level counter without inventing a job entry.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("LLM call count must be a positive integer")
        self._llm_calls += count
        self._publish_usage()

    def _publish_usage(self) -> None:
        if self._usage_out is None:
            return
        self._usage_out.update(
            {
                "llm_calls": max(self._llm_calls, int((self._usage_out or {}).get("llm_calls", 0))),
                "prompt_tokens": sum(result.prompt_tokens for result in self._job_results.values()),
                "completion_tokens": sum(result.completion_tokens for result in self._job_results.values()),
                "total_tokens": sum(result.total_tokens for result in self._job_results.values()),
            }
        )

    @property
    def input_shape_version(self) -> str:
        """Shape version the runner stamps on the reserved input object."""
        return self.definition.input_shape.version if self.definition.input_shape is not None else "1"

    def cancel(self) -> None:
        """Request cancellation; running jobs finish, no new job starts."""
        self._cancel_event.set()

    def job(self, job_id: str) -> CompiledJob:
        """Return the compiled job or raise a configuration error naming it."""
        try:
            return self.definition.jobs[job_id]
        except KeyError:
            raise KeyError(f"pipeline {self.definition.name!r} has no job {job_id!r}") from None

    def aliases_for(self, job_id: str) -> Mapping[str, str]:
        """Aliases registered by one job, delegated to the graph."""
        return self.graph.aliases_for(job_id)

    # ── Limits, deadline, cancellation ───────────────────────

    def check_limits(self) -> None:
        """Raise when a hard cap is already exhausted.

        Objects, relations and bytes are also enforced by the graph on write;
        this check fails the run before the next job starts and adds the
        counters the graph does not own (LLM calls).
        """
        if self.llm_calls > self.limits.max_llm_calls:
            raise OrchidPipelineLimitError(
                f"pipeline {self.definition.name!r} used {self._llm_calls} LLM calls, "
                f"exceeding max_llm_calls={self.limits.max_llm_calls}"
            )
        stats = self.graph.stats()
        for name, actual, maximum in (
            ("max_objects", stats["objects"], self.limits.max_objects),
            ("max_relations", stats["relations"], self.limits.max_relations),
            ("max_graph_bytes", stats["bytes"], self.limits.max_graph_bytes),
        ):
            if actual > maximum:
                raise OrchidPipelineLimitError(
                    f"pipeline {self.definition.name!r} is at {actual}, exceeding {name}={maximum}"
                )

    def check_deadline(self) -> None:
        """Raise :class:`OrchidPipelineTimeoutError` when the run deadline passed."""
        if self.deadline is not None and self._clock() >= self.deadline:
            raise OrchidPipelineTimeoutError(
                f"pipeline {self.definition.name!r} exceeded its deadline at {self.deadline.isoformat()}"
            )

    # ── Mapping context ──────────────────────────────────────

    # ORCHID-MEM: 01a1ef68-3f1b-7d3a-bbda-7899dc15d809 — Session mapping context accepts the deterministic summary for final output projection.
    def mapping_context(self, *, summary: str = "") -> OrchidPipelineMappingContext:
        """Build the documented render context from the current run state.

        ``summary`` is normally empty while jobs run. Final output projection
        passes the already computed deterministic summary so a mapping field can
        read it (spec §7.12.4); template/LLM summaries are produced after the
        output and therefore reject a ``summary`` reference at build time.
        """
        return build_mapping_context(
            input=self.input,
            run=self.run,
            jobs={job_id: result.to_dict() for job_id, result in self._job_results.items()},
            graph=self.graph.view(),
            aliases=self.graph.aliases,
            summary=summary,
        )

    # ORCHID-MEM: 01a1f2c5-f56c-7c74-941d-0606deab0a24 — Per-map semaphore; bounds map item concurrency per map job within the run's max_parallel.
    def map_semaphore(self, map_id: str, limit: int) -> asyncio.Semaphore:
        """Return the run-local semaphore for one map job.

        The coordinator creates it before fan-out so every item branch shares
        the same limit; the value is capped by the run's ``max_parallel``.
        """
        if not isinstance(map_id, str) or not map_id:
            raise ValueError("map_id must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("map concurrency must be a positive integer")
        semaphore = self._map_semaphores.get(map_id)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            self._map_semaphores[map_id] = semaphore
        return semaphore

    # ── Commits ──────────────────────────────────────────────

    # ORCHID-MEM: 01a1f2c5-f5a1-7365-98c9-d9e5c237b76b — Item-context commits; allowlist ids are rendered against the map item context.
    async def commit(
        self,
        job_id: str,
        mutations: Sequence[Any] = (),
        *,
        aliases: Mapping[str, str] | None = None,
        context: Any | None = None,
        extra_objects: Sequence[str] | None = None,
        extra_relations: Sequence[str] | None = None,
    ) -> DataGraphCommit:
        """Commit one job's mutations atomically, after the allowlist check.

        The allowlist is the job's declared ``writes``/``links`` rendered
        against ``context`` (the map item context for an item commit) or the
        current run context. A job that declares neither may write dynamically
        unless its config sets ``allow_dynamic_writes: false``. Everything
        happens under the session lock so two concurrent jobs cannot interleave
        a read of the allowlist with another job's commit.
        """
        job = self.job(job_id)
        async with self._lock:
            allow_objects, allow_relations = self._write_allowlist(job, context)
            if extra_objects and allow_objects is not None:
                allow_objects = set(allow_objects) | set(extra_objects)
            if extra_relations and allow_relations is not None:
                allow_relations = set(allow_relations) | set(extra_relations)
            mutation_list = tuple(mutations)
            for mutation in mutation_list:
                relation = getattr(mutation, "relation", None)
                if relation is not None:
                    if allow_relations is not None and relation.id not in allow_relations:
                        raise OrchidPipelineWriteNotAllowedError(
                            f"job {job_id!r} may not write relation {relation.id!r}; declared relations: "
                            f"{sorted(allow_relations) or '(none)'}"
                        )
                elif allow_objects is not None and mutation.object_id not in allow_objects:
                    raise OrchidPipelineWriteNotAllowedError(
                        f"job {job_id!r} may not write object {mutation.object_id!r}; declared objects: "
                        f"{sorted(allow_objects) or '(none)'}"
                    )

            alias_map = dict(aliases or {})
            declared_aliases = set(job.object_aliases)
            for alias in alias_map:
                if alias not in declared_aliases:
                    raise OrchidPipelineWriteNotAllowedError(
                        f"job {job_id!r} may not register alias {alias!r}; declared aliases: "
                        f"{sorted(declared_aliases) or '(none)'}"
                    )

            commit = self.graph.commit(job_id, mutation_list, aliases=alias_map)
            self._objects_written += len(commit.object_ids)
            self._relations_written += len(commit.relation_ids)
            # Capture the revision under the lock: a concurrent commit could
            # otherwise change it before the events are emitted.
            written = self._written_snapshot(commit)

        for object_id, object_type, revision in written:
            await self.emit(
                "pipeline.object.written",
                job=job_id,
                object_id=object_id,
                type=object_type,
                revision=revision,
            )
        return commit

    def _write_allowlist(self, job: CompiledJob, context: Any | None = None) -> tuple[set[str] | None, set[str] | None]:
        """Render the declared writes/links into concrete ids.

        ``None`` means "no restriction" and is only produced for a job that
        declares neither writes nor links and allows dynamic writes.
        """
        if not job.writes and not job.links:
            return (None, None) if job.allow_dynamic_writes else (set(), set())

        context = context if context is not None else self.mapping_context()
        path = f"job {job.id!r}"
        objects: set[str] = set()
        for write in job.writes:
            if write.when is not None and not evaluate_when(write.when, context, path=f"{path} write when"):
                continue
            objects.add(
                render_template(write.id, context, path=f"{path} write id", max_chars=self.limits.max_prompt_chars)
            )
        relations: set[str] = set()
        for link in job.links:
            if link.when is not None and not evaluate_when(link.when, context, path=f"{path} link when"):
                continue
            source = render_template(
                link.from_, context, path=f"{path} link from", max_chars=self.limits.max_prompt_chars
            )
            target = render_template(link.to, context, path=f"{path} link to", max_chars=self.limits.max_prompt_chars)
            relations.add(f"{link.type}:{source}:{target}")
        return objects, relations

    def _written_snapshot(self, commit: DataGraphCommit) -> list[tuple[str, str, int]]:
        """Bounded ``(id, type, revision)`` for the objects a commit wrote."""
        written: list[tuple[str, str, int]] = []
        for object_id in commit.object_ids:
            if len(written) >= _MAX_WRITTEN_EVENTS:
                break
            obj = self.graph.objects.get(object_id)
            if obj is not None:
                written.append((object_id, obj.type, obj.revision))
        return written

    # ── Job results ──────────────────────────────────────────

    async def job_started(self, job_id: str, attempt: int = 1) -> None:
        """Record and announce that a job attempt started."""
        job = self.job(job_id)
        await self.emit("pipeline.job.started", job=job_id, kind=job.kind, attempt=attempt)

    async def record_job_result(self, result: PipelineJobResult) -> None:
        """Store a job outcome, update counters and emit its lifecycle event.

        A failed job with the ``fail`` policy sets the fatal flag; the runner
        then skips every job that starts afterwards. Observer failures are
        logged and never change the outcome.
        """
        job = self.job(result.job_id)
        # Sanitize once at the boundary: the stored result feeds the job
        # report, the mapping context and the event, so redaction here covers
        # every consumer.
        result = replace(
            result,
            error=None if result.error is None else _bounded(result.error, _MAX_LOG_ERROR_CHARS),
            reason=_bounded(result.reason, _MAX_LOG_ERROR_CHARS),
            error_code=_bounded(result.error_code, 200),
        )
        self._job_results[result.job_id] = result
        self._llm_calls += result.llm_calls
        self._publish_usage()

        if result.status == "failed":
            self._errors.append(
                {
                    "job_id": result.job_id,
                    "kind": result.kind,
                    "code": result.error_code or "pipeline_step_error",
                    "message": _bounded(result.error or "", _MAX_LOG_ERROR_CHARS),
                    "retryable": result.retryable,
                }
            )
            self._errors = self._errors[:_MAX_ERRORS]
            if job.on_error == "fail":
                self._fatal = True
            await self.emit(
                "pipeline.job.failed",
                job=result.job_id,
                error_code=result.error_code or "pipeline_step_error",
                error=_bounded(result.error or "", _MAX_LOG_ERROR_CHARS),
                retryable=result.retryable,
            )
        elif result.status == "skipped":
            await self.emit("pipeline.job.skipped", job=result.job_id, reason=_bounded(result.reason, 200))
        else:
            await self.emit(
                "pipeline.job.finished",
                job=result.job_id,
                status=result.status,
                duration_ms=round(result.duration_ms, 3),
                objects_written=result.objects_written,
                llm_calls=result.llm_calls,
            )

        self._log_job(result)

    def _log_job(self, result: PipelineJobResult) -> None:
        self._logger.info(
            "pipeline=%s run=%s job=%s kind=%s status=%s duration_ms=%.3f objects_written=%d "
            "relations_written=%d llm_calls=%d error_code=%s",
            self.definition.name,
            self.run.run_id,
            result.job_id,
            result.kind,
            result.status,
            result.duration_ms,
            result.objects_written,
            result.relations_written,
            result.llm_calls,
            result.error_code or "-",
        )

    # ── Events and snapshots ─────────────────────────────────

    async def emit(self, event_type: str, **data: Any) -> None:
        """Emit one bounded event; a failing observer is logged and swallowed."""
        if self._observer is None:
            return
        event = OrchidPipelineEvent(type=event_type, run_id=self.run.run_id, data=data)
        try:
            await self._observer.emit(event)
        except Exception as exc:  # observability must never break a run
            self._logger.warning(
                "pipeline observer failed on %s: %s",
                event_type,
                _bounded(str(exc), _MAX_LOG_ERROR_CHARS),
            )

    def snapshot(self) -> dict[str, Any]:
        """Serialized graph snapshot (``{version, objects, relations}``)."""
        return self.graph.snapshot()

    def stats(self) -> dict[str, Any]:
        """Graph statistics plus run counters for the final result."""
        stats = self.graph.stats()
        stats.update(
            {
                "run_id": self.run.run_id,
                "pipeline": self.definition.name,
                "llm_calls": max(self._llm_calls, int((self._usage_out or {}).get("llm_calls", 0))),
                "prompt_tokens": sum(result.prompt_tokens for result in self._job_results.values()),
                "completion_tokens": sum(result.completion_tokens for result in self._job_results.values()),
                "total_tokens": sum(result.total_tokens for result in self._job_results.values()),
                "jobs": len(self._job_results),
                "errors": len(self._errors),
                "objects_written": self._objects_written,
                "relations_written": self._relations_written,
                "fatal": self._fatal,
                "cancelled": self.cancelled,
            }
        )
        return stats


# ORCHID-MEM: 01a1eb1b-d9ce-7e61-81d3-075d339eed99 — Concrete step context; run identity, graph view, injected tool caller and signal emitter, deterministic ids.
class PipelineStepContext:
    """Concrete step context handed to ``OrchidPipelineStep.execute``.

    Collaborators are injected: the session does not know how to call a tool or
    emit a signal, so the runner supplies ``tool_caller`` and ``signal_emitter``.
    Both raise a precise pipeline error when absent instead of failing with an
    ``AttributeError`` inside operator code.
    """

    def __init__(
        self,
        *,
        run: OrchidPipelineRunContext,
        job: OrchidPipelineJobContext,
        input: Mapping[str, Any],
        graph: Any,
        auth: Any,
        chat_model: Any | None = None,
        reader: Any | None = None,
        logger: logging.Logger | None = None,
        idempotency_key: str = "",
        tool_caller: Callable[..., Any] | None = None,
        signal_emitter: Any | None = None,
        child_runner: Any | None = None,
        usage_sink: dict[str, int] | None = None,
        llm_calls: int = 0,
        params: Mapping[str, Any] | None = None,
        pipeline: Any | None = None,
        mapping_context: Any | None = None,
    ) -> None:
        self.run = run
        self.job = job
        self.graph = graph
        self.auth = auth
        self.chat_model = chat_model
        self.reader = reader
        self.logger = logger or logging.getLogger(__name__)
        self.idempotency_key = idempotency_key or _default_idempotency_key(run)
        self.params = dict(params or {})
        self.input = MappingProxyType(copy.deepcopy(dict(input)))
        self.pipeline = pipeline
        self.mapping_context = mapping_context
        self._tool_caller = tool_caller
        self._signal_emitter = signal_emitter
        self.child_runner = child_runner
        self.usage_sink = usage_sink
        self.llm_calls = llm_calls

    def record_llm_call(self) -> None:
        """Publish one in-flight LLM call to the nested-run usage sink."""
        if self.usage_sink is not None:
            self.usage_sink["llm_calls"] = int(self.usage_sink.get("llm_calls", 0)) + 1

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a built-in tool with the run's auth injected."""
        if self._tool_caller is None:
            raise OrchidPipelineConfigError(
                f"pipeline {self.run.pipeline!r} has no built-in tool caller; tool {name!r} is unavailable"
            )
        return await self._tool_caller(name, auth_context=self.auth, **kwargs)

    async def emit_signal(self, envelope: Any) -> Any:
        """Emit a signal envelope; raises when events are disabled."""
        if self._signal_emitter is None:
            raise OrchidPipelineEventsDisabledError(
                f"pipeline {self.run.pipeline!r} cannot emit a signal: the events layer is disabled"
            )
        return await self._signal_emitter.emit(envelope)

    def stable_id(self, *parts: str) -> str:
        """Derive a deterministic id from the run namespace and ``parts``."""
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"orchid:pipeline:{self.run.pipeline}:{self.run.run_id}")
        return str(uuid.uuid5(namespace, ":".join(str(part) for part in parts)))


_TOKEN_PREFIX_RE = re.compile(r"(?i)^(gh[pousr]_|github_pat_|sk-|xox[a-z]-|akia|ya29\.|eyj)")
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~+/=-]{20,}$")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)(\S+)")
_BASIC_RE = re.compile(r"(?i)(authorization\s*[:=]\s*basic\s+)(\S+)")
_AUTH_HEADER_RE = re.compile(r"(?i)(authorization\s*[:=]\s*)(?!basic\b)(\S+)")


def _is_token_value(value: str) -> bool:
    """Heuristic for a bearer credential: known prefix, or a long opaque token.

    The heuristic protects the common token families and any long value with a
    separator or a digit, while leaving ordinary prose (``Bearer
    authentication failed``) readable in an error message.
    """
    if _TOKEN_PREFIX_RE.match(value):
        return True
    if len(value) < 20 or not _OPAQUE_TOKEN_RE.match(value):
        return False
    # A digit is the strongest signal; a long mixed-case alpha run is a base64
    # fragment. Requiring one of the two keeps hyphenated prose readable.
    return any(char.isdigit() for char in value) or (value.isalpha() and value != value.lower())


def _is_header_credential(value: str) -> bool:
    """Heuristic for an authorization-header value: a token, not a word."""
    unquoted = value.strip("'\"")
    return bool(len(unquoted) >= 8 and (_TOKEN_PREFIX_RE.match(unquoted) or any(char.isdigit() for char in unquoted)))


def _redact(message: str) -> str:
    """Remove credential-shaped bearer/basic/authorization values from a message.

    A step error is operator-authored, but it can wrap a transport exception
    whose message contains the request headers. R-016 requires that no token
    reach an error record, an event or a log line.
    """

    def _keep_or_redact(match: re.Match[str]) -> str:
        value = match.group(2)
        return match.group(1) + ("[redacted]" if _is_token_value(value.strip("'\"")) else value)

    def _keep_or_redact_header(match: re.Match[str]) -> str:
        value = match.group(2)
        return match.group(1) + ("[redacted]" if _is_header_credential(value) else value)

    message = _BASIC_RE.sub(lambda match: match.group(1) + "[redacted]", message)
    message = _BEARER_RE.sub(_keep_or_redact, message)
    return _AUTH_HEADER_RE.sub(_keep_or_redact_header, message)


def _default_idempotency_key(run: OrchidPipelineRunContext) -> str:
    """Derive the documented key: ``trigger:signal`` for Bloom, else ``run_id``."""
    if run.trigger_id and run.signal_id:
        return f"{run.trigger_id}:{run.signal_id}"
    return run.run_id


def _bounded(message: str, limit: int) -> str:
    redacted = _redact(message)
    if len(redacted) <= limit:
        return redacted
    return redacted[: limit - 1] + "…"
