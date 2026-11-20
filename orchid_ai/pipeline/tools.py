"""Pipeline tools for agentic use.

A pipeline becomes an ``OrchidTool`` only when its definition sets
``expose.as_tool: true``. The tool is a thin adapter: it reads the live auth and
chat context, calls the injected pipeline invoker (sync) or launcher (async),
and returns a bounded JSON payload. It never writes to chat storage — the parent
agent owns the conversation turn.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..core.agent import current_run_context
from ..core.pipeline import OrchidPipelineConfigError, OrchidPipelineError
from ..core.tool import (
    OrchidTool,
    OrchidToolInput,
    OrchidToolOutput,
    tool_approval_granted,
)
from .launcher import PipelineLauncher, PipelineLaunchResult
from .registry import OrchidPipelineToolSpec

__all__ = ["PipelineTool", "PipelineToolInvoker", "sanitize_tool_name"]

_logger = logging.getLogger(__name__)

PipelineToolInvoker = Callable[..., Awaitable[Any]]

_TOOL_NAME_MAX = 64
_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_VALID_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_STATUS_CODES = {
    "failed": "pipeline_failed",
    "partial": "pipeline_partial",
    "cancelled": "pipeline_cancelled",
}
_MAX_ERROR_CHARS = 500
_MIN_RESULT_CHARS = 160


def sanitize_tool_name(name: str) -> str:
    """Return a registered-tool-safe name (spec §8.2.1).

    Characters outside ``[A-Za-z0-9_-]`` become ``_``; a leading
    non-alphanumeric gets the ``pipeline_`` prefix, and the result is capped at
    64 characters. Empty input becomes ``pipeline``. Separators are preserved,
    matching the registry's validated tool-name rule.
    """
    candidate = _TOOL_NAME_RE.sub("_", str(name or ""))[:_TOOL_NAME_MAX]
    if not candidate:
        return "pipeline"
    if not candidate[0].isalnum():
        candidate = f"pipeline_{candidate}"[:_TOOL_NAME_MAX]
    return candidate


def _bounded_text(value: Any, *, limit: int = _MAX_ERROR_CHARS) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = text.replace("\x00", "\ufffd")
    return text[:limit]


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True, default=str))


def _bounded_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Trim a tool payload to ``max_chars`` serialized characters.

    Required fields (pipeline, run id, status and ``error.code``) survive the
    reduction; optional output, graph stats, duration and summary are dropped
    first, then the error message is shortened.
    """
    if _json_size(payload) <= max_chars:
        return payload

    trimmed = dict(payload)
    trimmed["truncated"] = True
    for key, empty in (("output", None), ("graph_stats", {}), ("duration_ms", None)):
        if trimmed.get(key) not in (None, {}, []):
            trimmed[key] = empty
            if _json_size(trimmed) <= max_chars:
                return trimmed

    summary = trimmed.get("summary")
    if isinstance(summary, str) and summary:
        without_summary = {**trimmed, "summary": ""}
        available = max_chars - _json_size(without_summary)
        trimmed["summary"] = _bounded_text(summary, limit=max(0, available))
        if _json_size(trimmed) <= max_chars:
            return trimmed
    trimmed["summary"] = ""
    if _json_size(trimmed) <= max_chars:
        return trimmed

    error = trimmed.get("error")
    if isinstance(error, dict):
        message = error.get("message", "")
        for limit in (200, 100, 50, 0):
            trimmed["error"] = {**error, "message": _bounded_text(message, limit=limit)}
            if _json_size(trimmed) <= max_chars:
                return trimmed
        trimmed["error"] = {"code": error.get("code", "pipeline_error")}
        if _json_size(trimmed) <= max_chars:
            return trimmed

    return _required_payload(trimmed, max_chars)


def _required_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Last-resort reducer that keeps the documented required fields.

    The run id and status are mandatory; the pipeline name and error code are
    kept when they fit, with the pipeline name shortened to its budget. The
    caller's cap is raised to a workable minimum in ``PipelineTool``, so this
    function only has to handle a genuinely small budget.
    """
    required: dict[str, Any] = {
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", "failed"),
        "truncated": True,
    }
    if payload.get("signal_id"):
        required["signal_id"] = payload["signal_id"]
    error = payload.get("error")
    if isinstance(error, dict):
        required["error"] = {"code": error.get("code", "pipeline_error")}

    pipeline = payload.get("pipeline", "")
    candidate = {"pipeline": pipeline, **required}
    if _json_size(candidate) <= max_chars:
        return candidate

    budget = max_chars - _json_size({**required, "pipeline": ""})
    if budget > 2:
        candidate = {"pipeline": str(pipeline)[: max(0, budget - 2)], **required}
        if _json_size(candidate) <= max_chars:
            return candidate

    if "error" in required:
        candidate = {"pipeline": "", **required}
        if _json_size(candidate) <= max_chars:
            return candidate
        required.pop("error", None)

    candidate = {"pipeline": "", **required}
    if _json_size(candidate) <= max_chars:
        return candidate
    return {"run_id": required["run_id"], "truncated": True}


def _failed(
    code: str,
    message: str,
    *,
    pipeline: str = "",
    max_chars: int = 8000,
) -> dict[str, Any]:
    payload = {
        "pipeline": pipeline,
        "run_id": "",
        "status": "failed",
        "summary": "",
        "output": None,
        "graph_stats": {},
        "duration_ms": None,
        "truncated": False,
        "error": {"code": code, "message": _bounded_text(message)},
    }
    return _bounded_payload(payload, max_chars)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _tool_description(spec: OrchidPipelineToolSpec) -> str:
    base = (spec.description or f"Run the {spec.pipeline!r} pipeline.").strip()
    input_names = sorted((spec.input_schema or {}).get("properties") or {})
    if input_names:
        base += f" Input: {', '.join(input_names)}."
    guidance = (
        " Returns a structured result and a human-readable summary; use the "
        "summary to answer the user and do not dump the raw JSON."
    )
    if "use the summary" in base.lower():
        return base
    return base + guidance


# ORCHID-MEM: 01a1f37c-2062-74ea-a970-66b53d1a60c7 — Pipeline agent tool; live auth/chat context, bounded sync/async result and no chat writes.
class PipelineTool(OrchidTool):
    """Expose one compiled pipeline as a built-in agent tool."""

    def __init__(
        self,
        *,
        spec: OrchidPipelineToolSpec,
        invoker: PipelineToolInvoker,
        launcher: PipelineLauncher | None = None,
        default_timeout: float | None = None,
    ) -> None:
        input_properties = set((spec.input_schema or {}).get("properties") or {})
        reserved = input_properties & {
            "mode",
            "timeout_seconds",
            "context",
            "auth_context",
            "content_sources",
        }
        if reserved:
            raise OrchidPipelineConfigError(
                f"pipeline {spec.pipeline!r} input declares reserved tool parameter(s) "
                f"{sorted(reserved)}; the pipeline tool owns framework parameters"
            )
        self.pipeline = spec.pipeline
        self.name = (
            spec.name
            if _VALID_TOOL_NAME_RE.fullmatch(spec.name or "")
            else sanitize_tool_name(spec.name or spec.pipeline)
        )
        self.description = _tool_description(spec)
        self.parameters_schema = _merge_parameters_schema(spec.input_schema)
        self.requires_approval = bool(spec.requires_approval)
        self.parallel_safe = bool(spec.parallel_safe)
        self.inject_to_rag = False
        self.require_role = spec.require_role
        # Below this the required result fields cannot be represented; the
        # configured cap is a context budget, not a security boundary.
        self._max_tool_result_chars = max(_MIN_RESULT_CHARS, int(spec.max_tool_result_chars))
        self._invoker = invoker
        self._launcher = launcher
        self._default_timeout = default_timeout

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        params = dict(tool_input.parameters or {})
        mode = str(params.pop("mode", "sync") or "sync").lower()
        raw_timeout = params.pop("timeout_seconds", None)
        timeout = self._resolve_timeout(raw_timeout)
        if timeout is _INVALID:
            return self._error("pipeline_input_error", "timeout_seconds must be a positive number")

        auth = tool_input.auth_context
        if auth is None:
            return self._error(
                "pipeline_auth_error",
                f"pipeline {self.pipeline!r} requires an authenticated caller",
            )

        allowed, role_error = self._check_role(auth)
        if not allowed:
            return self._error("pipeline_forbidden", role_error)

        run_context = current_run_context()
        chat_id = run_context.chat_id if run_context else None
        correlation_id = run_context.correlation_id if run_context else None

        if self.requires_approval and not self._approval_available():
            return self._error(
                "pipeline_approval_required",
                (
                    f"pipeline {self.pipeline!r} requires human approval, but no chat/approval "
                    "handler is available. Run it from a chat with HITL enabled or disable "
                    "expose.requires_approval."
                ),
            )

        if mode == "async":
            return await self._invoke_async(
                params,
                auth=auth,
                chat_id=chat_id,
                correlation_id=correlation_id,
                message_id=run_context.message_id if run_context else None,
                timeout=timeout,
            )
        if mode != "sync":
            return self._error("pipeline_input_error", f"unknown pipeline tool mode {mode!r}")

        try:
            result = await self._invoker(
                self.pipeline,
                params,
                auth=auth,
                chat_id=chat_id,
                correlation_id=correlation_id,
                timeout=timeout,
            )
        except OrchidPipelineError as exc:
            return self._error(exc.code, str(exc))
        except Exception as exc:
            _logger.exception("pipeline tool %s failed", self.name)
            return self._error("pipeline_error", f"pipeline {self.pipeline!r} failed ({type(exc).__name__})")

        return OrchidToolOutput(result=self._result_payload(result))

    def _check_role(self, auth: Any) -> tuple[bool, str]:
        if not self.require_role:
            return True, ""
        roles = getattr(auth, "roles", None) or ()
        if self.require_role in roles:
            return True, ""
        return (
            False,
            f"pipeline {self.pipeline!r} requires role {self.require_role!r}",
        )

    def _approval_available(self) -> bool:
        # The agentic loop sets this marker only after its HITL interrupt is
        # approved. Model-supplied tool context is never an approval signal.
        return tool_approval_granted(self.name)

    def _resolve_timeout(self, raw: Any) -> float | None | object:
        if raw is None:
            return self._default_timeout
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return _INVALID
        if timeout <= 0:
            return _INVALID
        return timeout

    async def _invoke_async(
        self,
        params: dict[str, Any],
        *,
        auth: Any,
        chat_id: str | None,
        correlation_id: str | None,
        message_id: str | None,
        timeout: float | None,
    ) -> OrchidToolOutput:
        if self._launcher is None:
            return self._error(
                "pipeline_async_unavailable",
                (
                    f"pipeline {self.pipeline!r} has no async launcher; enable events and "
                    "configure pipelines.<name>.async"
                ),
            )
        binding = None
        if chat_id:
            binding = {
                "chat_id": chat_id,
                "mode": "append_final_message",
                "on_failure": "post_error",
                "source_message_id": message_id,
            }
        try:
            launched = await self._launcher.launch(
                self.pipeline,
                params,
                auth=auth,
                chat_binding=binding,
                correlation_id=correlation_id,
                timeout_seconds=timeout,
            )
        except Exception as exc:
            _logger.exception("pipeline launcher failed for %s", self.pipeline)
            return self._error(
                "pipeline_async_unavailable",
                f"pipeline {self.pipeline!r} could not be started ({type(exc).__name__})",
            )

        if isinstance(launched, PipelineLaunchResult):
            payload = launched.to_dict()
        elif isinstance(launched, Mapping):
            payload = dict(launched)
        else:  # pragma: no cover — launcher contract returns a result
            return self._error("pipeline_async_unavailable", "the async launcher returned no result")
        payload.setdefault("truncated", False)
        return OrchidToolOutput(result=_bounded_payload(payload, self._max_tool_result_chars))

    def _result_payload(self, result: Any) -> dict[str, Any]:
        status = str(_value(result, "status", "failed") or "failed")
        if status not in {"succeeded", "failed", "partial", "cancelled"}:
            status = "failed"
        payload: dict[str, Any] = {
            "pipeline": str(_value(result, "pipeline", self.pipeline) or self.pipeline),
            "run_id": str(_value(result, "run_id", "") or ""),
            "status": status,
            "summary": _value(result, "summary", "") or "",
            "output": _value(result, "output", None),
            "graph_stats": dict(_value(result, "graph_stats", {}) or {}),
            "duration_ms": _value(result, "duration_ms", None),
            "truncated": False,
        }
        if status != "succeeded":
            error = _value(result, "error", None)
            payload["error"] = {
                "code": _STATUS_CODES.get(status, "pipeline_error"),
                "message": _bounded_text(error or f"pipeline finished with status {status}"),
            }
        if not isinstance(payload["summary"], str):
            payload["summary"] = _bounded_text(payload["summary"])
        if not payload["summary"]:
            payload["summary"] = (
                f"Pipeline {self.pipeline!r} completed successfully."
                if status == "succeeded"
                else f"Pipeline {self.pipeline!r} finished with status {status}."
            )
        return _bounded_payload(payload, self._max_tool_result_chars)

    def _error(self, code: str, message: str) -> OrchidToolOutput:
        return OrchidToolOutput(
            result=_failed(
                code,
                message,
                pipeline=self.pipeline,
                max_chars=self._max_tool_result_chars,
            )
        )


_INVALID = object()


def _merge_parameters_schema(input_schema: Mapping[str, Any] | None) -> dict[str, Any]:
    schema = dict(input_schema or {})
    properties = dict(schema.get("properties") or {})
    properties["mode"] = {
        "type": "string",
        "enum": ["sync", "async"],
        "default": "sync",
        "description": "Run inline and return the result, or start a Bloom job and return accepted.",
    }
    properties["timeout_seconds"] = {
        "type": "number",
        "minimum": 0.1,
        "description": "Optional wall-clock timeout for the pipeline run.",
    }
    merged = {
        **schema,
        "type": "object",
        "properties": properties,
    }
    return merged
