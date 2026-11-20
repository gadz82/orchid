"""Bloom pipeline runner: execute a pipeline JobSpec and store its result.

The runner is target-neutral about *how* the pipeline runs: the composition
root injects an object exposing ``run(name, input, ...)``. This module owns the
JobRun contract, the shared chat-binding policy, result projection and the
optional graph persistence cap. It never puts a bearer token in the run result.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from ...core.events.errors import (
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
    JobRunnerError,
)
from ...core.events.job import JobRun, JobStatus
from ...core.events.runner import OrchidJobRunner
from .chat_binding import ChatBindingDeliverer

_logger = logging.getLogger(__name__)

MAX_PERSISTED_GRAPH_BYTES = 1_000_000
_STATUSES = frozenset({"succeeded", "failed", "partial", "cancelled"})
_MAX_TEXT_CHARS = 2000


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _check_json_text(value: Any, *, path: str = "value") -> None:
    """Reject NUL characters that PostgreSQL JSONB cannot store."""
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"{path} contains a NUL character")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_json_text(key, path=f"{path} key")
            _check_json_text(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_json_text(item, path=f"{path}[{index}]")


def _sanitize_text(value: Any, *, fallback: str = "") -> str:
    """Return bounded text that PostgreSQL TEXT/JSONB can always store."""
    text = value if isinstance(value, str) else fallback
    return text.replace("\x00", "\ufffd")[:_MAX_TEXT_CHARS]


def _json_safe(value: Any) -> Any:
    """Return a JSON-native copy so the job store cannot fail on encoding."""
    _check_json_text(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


# ORCHID-MEM: 01a1f342-64a8-7938-ace4-132c318e4131 — Pipeline job runner; runs a pipeline JobSpec, shapes the Bloom result and shares chat delivery.
class PipelineJobRunner(OrchidJobRunner):
    """Run a ``target: pipeline`` JobRun through an injected pipeline runner."""

    def __init__(
        self,
        *,
        pipeline_runner: Any,
        chat_deliverer: ChatBindingDeliverer | None = None,
        chat_storage: Any | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
        max_graph_bytes: int = MAX_PERSISTED_GRAPH_BYTES,
    ) -> None:
        self._pipeline_runner = pipeline_runner
        self._clock = clock or _now
        self._max_graph_bytes = max(1, int(max_graph_bytes))
        self._deliverer = chat_deliverer or ChatBindingDeliverer(
            chat_storage=chat_storage,
            clock=self._clock,
        )

    async def run(self, run: JobRun, *, auth: Any) -> None:
        try:
            binding = await self._deliverer.resolve_binding(run, auth)
        except (ChatBindingTargetNotFoundError, ChatBindingForbiddenError) as exc:
            _logger.warning(
                "chat-binding rejected for run %s (trigger=%s): %s",
                run.run_id,
                run.spec.trigger_id,
                exc,
            )
            run.status = JobStatus.FAILED
            run.error = repr(exc)
            run.finished_at = self._clock()
            raise JobRunnerError(repr(exc), retryable=False) from exc

        try:
            raw = await self._pipeline_runner.run(
                run.spec.pipeline_name or "",
                dict(run.spec.pipeline_input or {}),
                auth=auth,
                mode="bloom",
                correlation_id=run.spec.correlation_id,
                chat_id=binding["chat_id"] if binding is not None else None,
                include_graph=run.spec.persist_graph,
            )
        except JobRunnerError as exc:
            await self._deliverer.deliver_failure(binding, run, exc)
            raise
        except Exception as exc:
            _logger.exception(
                "pipeline invocation failed for run %s (trigger=%s)",
                run.run_id,
                run.spec.trigger_id,
            )
            await self._deliverer.deliver_failure(binding, run, exc)
            raise JobRunnerError(
                repr(exc),
                retryable=bool(getattr(exc, "retryable", False)),
            ) from exc

        try:
            result, graph, pipeline_error = self._normalize_result(run, raw)
        except JobRunnerError as exc:
            await self._deliverer.deliver_failure(binding, run, exc)
            raise
        except Exception as exc:
            error = JobRunnerError(repr(exc), retryable=False)
            await self._deliverer.deliver_failure(binding, run, error)
            raise error from exc

        status = result["status"]
        if status != "succeeded":
            # A graph-cap problem must never mask the pipeline's own failure.
            if run.spec.persist_graph:
                prepared, graph_error = self._prepare_graph(graph)
                if graph_error is None:
                    result["graph"] = prepared
                else:
                    _logger.warning(
                        "dropping graph for failed run %s: %s",
                        run.run_id,
                        graph_error,
                    )
            if not result["final_response"]:
                result["final_response"] = f"Pipeline {run.spec.pipeline_name!r} finished with status {status}"
            message = pipeline_error or f"pipeline finished with status {status}"
            return await self._fail_run(binding, run, result, message)

        if not result["final_response"]:
            result["final_response"] = f"Pipeline {run.spec.pipeline_name!r} completed successfully."

        if run.spec.persist_graph:
            prepared, graph_error = self._prepare_graph(graph)
            if graph_error is not None:
                # Negotiation decision A: the audit snapshot is not silently
                # lost; the Bloom run fails after the pipeline succeeded.
                return await self._fail_run(binding, run, result, graph_error)
            result["graph"] = prepared
        try:
            result = _json_safe(result)
        except (TypeError, ValueError) as exc:
            message = f"pipeline result is not JSON-serializable: {type(exc).__name__}"
            failure_result = {
                **result,
                "final_response": (
                    f"Pipeline {run.spec.pipeline_name!r} completed but its result could not be stored."
                ),
                "status": "failed",
            }
            return await self._fail_run(binding, run, failure_result, message)

        run.result = result
        run.status = JobStatus.SUCCEEDED
        run.finished_at = self._clock()
        await self._deliverer.deliver_success(binding, run, result)

    async def _fail_run(
        self,
        binding: dict[str, Any] | None,
        run: JobRun,
        result: dict[str, Any],
        message: str,
    ) -> None:
        result["final_response"] = _sanitize_text(
            result.get("final_response"),
            fallback=f"Pipeline {run.spec.pipeline_name!r} failed",
        )
        try:
            result = _json_safe(result)
        except (TypeError, ValueError):
            status = result.get("status")
            if status not in _STATUSES:
                status = "failed"
            result = {
                "final_response": result["final_response"],
                "pipeline": run.spec.pipeline_name,
                "pipeline_run_id": "",
                "status": status,
                "output": None,
                "graph_stats": {},
            }
        run.result = result
        run.status = JobStatus.FAILED
        run.error = _sanitize_text(message, fallback="pipeline failed")
        run.finished_at = self._clock()
        await self._deliverer.deliver_failure(binding, run, JobRunnerError(message, retryable=False))

    def _prepare_graph(self, graph: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not graph:
            return {}, None
        try:
            _check_json_text(graph, path="graph")
            encoded = json.dumps(graph, ensure_ascii=False, allow_nan=False, default=str)
        except (TypeError, ValueError) as exc:
            return None, f"pipeline graph is not JSON-serializable: {type(exc).__name__}"
        size = len(encoded.encode("utf-8"))
        if size > self._max_graph_bytes:
            message = f"pipeline graph is {size} bytes, exceeding the {self._max_graph_bytes}-byte persistence cap"
            return None, message
        return json.loads(encoded), None

    def _normalize_result(
        self,
        run: JobRun,
        raw: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        graph: Mapping[str, Any] = {}
        if hasattr(raw, "to_dict"):
            try:
                payload = raw.to_dict(include_graph=False)
            except TypeError:
                payload = raw.to_dict()
            graph = getattr(raw, "graph", {}) or {}
        elif isinstance(raw, Mapping):
            payload = dict(raw)
            graph = payload.pop("graph", {}) or {}
        elif hasattr(raw, "status") and hasattr(raw, "run_id"):
            # Duck-typed pipeline result without ``to_dict`` so the events
            # layer does not import the pipeline runtime.
            payload = {
                "final_response": getattr(raw, "summary", "") or "",
                "pipeline": getattr(raw, "pipeline", run.spec.pipeline_name),
                "pipeline_run_id": getattr(raw, "run_id", ""),
                "status": getattr(raw, "status", ""),
                "output": getattr(raw, "output", None),
                "graph_stats": getattr(raw, "graph_stats", None),
                "error": getattr(raw, "error", None),
            }
            graph = getattr(raw, "graph", {}) or {}
        else:
            raise JobRunnerError(
                f"pipeline runner returned {type(raw).__name__}, expected a pipeline result or mapping",
                retryable=False,
            )

        if not isinstance(payload, Mapping):
            raise JobRunnerError(
                "pipeline runner result must be a mapping",
                retryable=False,
            )

        raw_status = payload.get("status")
        if not raw_status:
            status = "failed"
            pipeline_error = "pipeline runner returned no status"
        elif str(raw_status) not in _STATUSES:
            status = "failed"
            pipeline_error = f"pipeline runner returned unknown status {raw_status!r}"
        else:
            status = str(raw_status)
            pipeline_error = payload.get("error")
            if pipeline_error is not None:
                pipeline_error = str(pipeline_error)

        final_response = payload.get("final_response")
        if not isinstance(final_response, str) or not final_response:
            summary = payload.get("summary")
            final_response = summary if isinstance(summary, str) else ""

        pipeline_run_id = payload.get("pipeline_run_id") or payload.get("run_id") or ""
        graph_stats = payload.get("graph_stats")
        result = {
            "final_response": final_response,
            "pipeline": payload.get("pipeline") or run.spec.pipeline_name or "",
            "pipeline_run_id": str(pipeline_run_id),
            "status": status,
            "output": payload.get("output"),
            "graph_stats": dict(graph_stats) if isinstance(graph_stats, Mapping) else {},
        }
        return result, dict(graph) if isinstance(graph, Mapping) else {}, pipeline_error
