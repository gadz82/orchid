"""Signal-based async pipeline launcher.

The launcher is the narrow adapter between an exposed pipeline and Pollen:
it verifies that events are enabled and that the pipeline's async binding names
a real trigger for this pipeline, then emits one ``SignalEnvelope`` carrying the
pipeline input and identity claim. It never places a bearer token in the signal
and never runs the pipeline in-process.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..config.schema_events import OrchidEventsConfig, OrchidTriggerConfig
from ..core.events.signal import SignalEnvelope
from .registry import PipelineRegistry

__all__ = ["PipelineLaunchResult", "PipelineLauncher"]

_ASYNC_UNAVAILABLE = "pipeline_async_unavailable"
_CHAT_BINDING_KEYS = frozenset({"chat_id", "mode", "on_failure", "source_message_id"})


@dataclass(frozen=True, slots=True)
class PipelineLaunchResult:
    """Outcome of one async launch request."""

    status: str
    pipeline: str
    signal_id: str = ""
    message: str = ""
    deduplicated: bool = False
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "pipeline": self.pipeline,
            "signal_id": self.signal_id,
            "message": self.message,
            "deduplicated": self.deduplicated,
        }
        if self.error is not None:
            payload["error"] = dict(self.error)
        return payload


# ORCHID-MEM: 01a1f37c-2098-7b5a-922c-14d72b3a644d — Async pipeline launcher; validates events/trigger and emits an identity-only signal.
class PipelineLauncher:
    """Start a pipeline through Bloom when its async binding is available."""

    def __init__(
        self,
        *,
        registry: PipelineRegistry,
        events_config: OrchidEventsConfig | None,
        signal_emitter: Any | None = None,
        signal_emitter_getter: Callable[[], Any] | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
        source: str = "internal:pipeline_tool",
    ) -> None:
        self._registry = registry
        self._events_config = events_config
        self._signal_emitter = signal_emitter
        self._signal_emitter_getter = signal_emitter_getter
        self._clock = clock or (lambda: _dt.datetime.now(tz=_dt.UTC))
        self._source = source

    async def launch(
        self,
        name: str,
        input: Mapping[str, Any],
        *,
        auth: Any,
        chat_binding: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        timeout_seconds: float | None = None,
        dedupe_key: str | None = None,
    ) -> PipelineLaunchResult:
        """Validate the async binding and emit the launch signal.

        ``timeout_seconds`` is accepted for the tool/API contract; the pipeline
        run's own limits still decide its timeout when Bloom executes it.
        """
        if not isinstance(input, Mapping):
            return PipelineLaunchResult(
                status="failed",
                pipeline=name if isinstance(name, str) else "",
                error={
                    "code": "pipeline_input_error",
                    "message": "pipeline input must be an object",
                },
            )
        definition = self._registry.get(name) if isinstance(name, str) else None
        if definition is None:
            return self._unavailable(name, f"pipeline {name!r} is not configured")

        async_config = definition.definition.async_
        if not async_config.enabled:
            return self._unavailable(name, f"pipeline {name!r} does not enable async execution")
        if self._events_config is None or not self._events_config.enabled:
            return self._unavailable(
                name,
                f"pipeline {name!r} cannot run asynchronously because events are disabled",
            )
        if async_config.mode != "signal":
            return self._unavailable(
                name,
                f"pipeline {name!r} uses async.mode={async_config.mode!r}; the launcher supports 'signal'",
            )

        trigger = _find_trigger(self._events_config, async_config.trigger_id)
        if trigger is None:
            return self._unavailable(
                name,
                f"async.trigger_id {async_config.trigger_id!r} does not exist in events.triggers",
            )
        if trigger.emits.pipeline != name:
            return self._unavailable(
                name,
                f"trigger {trigger.id!r} targets pipeline {trigger.emits.pipeline!r}, not {name!r}",
            )
        if async_config.signal and trigger.on.signal != async_config.signal:
            return self._unavailable(
                name,
                f"trigger {trigger.id!r} listens on {trigger.on.signal!r}, not {async_config.signal!r}",
            )
        if trigger.emits.input and input:
            return self._unavailable(
                name,
                f"trigger {trigger.id!r} declares a static emits.input; launched input would be ignored. "
                "Use emits.input_from: payload.input for an async trigger.",
            )

        emitter = self._emitter()
        if emitter is None:
            return self._unavailable(
                name,
                "the events runtime is not started; inject the signal emitter before launching",
            )
        if auth is None:
            return PipelineLaunchResult(
                status="failed",
                pipeline=name,
                error={
                    "code": "pipeline_auth_error",
                    "message": f"pipeline {name!r} requires an authenticated caller",
                },
            )

        respects_binding = bool(trigger.emits.respect_chat_binding) and (
            trigger.emits.identity.mode != "service_account"
        )
        cleaned_binding = _clean_chat_binding(chat_binding) if respects_binding else None
        envelope = SignalEnvelope(
            type=trigger.on.signal,
            payload={"pipeline": name, "input": dict(input)},
            source=f"{self._source}:{name}",
            occurred_at=self._clock(),
            tenant_key=getattr(auth, "tenant_key", "default"),
            user_id=getattr(auth, "user_id", None) or None,
            correlation_id=correlation_id,
            dedupe_key=dedupe_key,
            identity_claim=_identity_claim(trigger, auth),
            chat_binding=cleaned_binding,
        )
        try:
            ingested = await emitter.emit(envelope)
        except Exception as exc:
            return PipelineLaunchResult(
                status="failed",
                pipeline=name,
                error={
                    "code": "pipeline_async_emit_failed",
                    "message": f"pipeline {name!r} could not emit its launch signal ({type(exc).__name__})",
                },
            )

        message = (
            "Started; results will be posted in this chat."
            if cleaned_binding
            else "Started; results will be available in the run history."
        )
        return PipelineLaunchResult(
            status="accepted",
            pipeline=name,
            signal_id=str(getattr(ingested, "signal_id", "")),
            message=message,
            deduplicated=bool(getattr(ingested, "deduplicated", False)),
        )

    def _emitter(self) -> Any | None:
        if self._signal_emitter_getter is not None:
            return self._signal_emitter_getter()
        return self._signal_emitter

    def _unavailable(self, name: str, message: str) -> PipelineLaunchResult:
        return PipelineLaunchResult(
            status="failed",
            pipeline=name if isinstance(name, str) else "",
            error={"code": _ASYNC_UNAVAILABLE, "message": message},
        )


def _find_trigger(
    events_config: OrchidEventsConfig,
    trigger_id: str | None,
) -> OrchidTriggerConfig | None:
    if not trigger_id:
        return None
    return next((trigger for trigger in events_config.triggers if trigger.id == trigger_id), None)


def _identity_claim(trigger: OrchidTriggerConfig, auth: Any) -> dict[str, Any]:
    claim = dict(trigger.emits.identity.model_dump())
    if claim.get("mode") == "act_as_user":
        claim["user_id"] = getattr(auth, "user_id", "")
    return claim


def _clean_chat_binding(chat_binding: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the documented §25 keys; never forward arbitrary signal data."""
    if not chat_binding:
        return None
    cleaned = {key: chat_binding[key] for key in _CHAT_BINDING_KEYS if key in chat_binding}
    if not cleaned.get("chat_id"):
        return None
    return cleaned
