"""Async pipeline launcher: trigger validation and identity-only signal emission."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import sys
import types
import uuid

import pytest

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidEventsConfig,
    OrchidProcessorConfig,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
)
from orchid_ai.config.schema_pipeline import OrchidPipelinesConfig
from orchid_ai.core.events.signal import SignalIngestResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.launcher import PipelineLauncher
from orchid_ai.pipeline.registry import PipelineRegistry


@pytest.fixture
def launcher_steps(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("launcher_steps")

    def load(ctx):
        return {"ok": True}

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "launcher_steps", module)
    return "launcher_steps"


def _pipelines(
    *,
    async_enabled: bool = True,
    async_mode: str = "signal",
    async_signal: str = "pipeline.demo.requested",
    trigger_id: str = "demo-trigger",
) -> OrchidPipelinesConfig:
    return OrchidPipelinesConfig.model_validate(
        {
            "enabled": True,
            "definitions": {
                "demo": {
                    "description": "Demo pipeline",
                    "input": {"fields": {"account_id": {"type": "string", "required": True}}},
                    "jobs": {
                        "load": {
                            "kind": "deterministic",
                            "uses": "launcher_steps.load",
                        }
                    },
                    "async": {
                        "enabled": async_enabled,
                        "signal": async_signal,
                        "trigger_id": trigger_id,
                        "mode": async_mode,
                    },
                }
            },
        }
    )


def _registry(**kwargs) -> PipelineRegistry:
    return PipelineRegistry.from_config(_pipelines(**kwargs))


def _trigger(*, respect_chat_binding: bool = False, **overrides) -> OrchidTriggerConfig:
    values = {
        "id": "demo-trigger",
        "on": OrchidTriggerMatchConfig(signal="pipeline.demo.requested"),
        "emits": OrchidTriggerEmitConfig(
            pipeline="demo",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
            respect_chat_binding=respect_chat_binding,
        ),
    }
    values.update(overrides)
    return OrchidTriggerConfig(**values)


def _events(*, triggers: list[OrchidTriggerConfig] | None = None, enabled: bool = True) -> OrchidEventsConfig:
    return OrchidEventsConfig(
        enabled=enabled,
        processors=[
            OrchidProcessorConfig(
                class_path="orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor",
                poll_interval_ms=10_000,
            )
        ],
        triggers=triggers if triggers is not None else [_trigger()],
    )


class _Emitter:
    def __init__(self, *, signal_id: uuid.UUID | None = None, error: Exception | None = None) -> None:
        self.envelopes = []
        self.signal_id = signal_id or uuid.uuid4()
        self.error = error

    async def emit(self, envelope):
        self.envelopes.append(envelope)
        if self.error is not None:
            raise self.error
        return SignalIngestResult(signal_id=self.signal_id, queue_msg_id="q-1")


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="secret-token", tenant_key="t-1", user_id="u-1")


async def test_launch_emits_an_identity_only_signal(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(triggers=[_trigger(respect_chat_binding=True)]),
        signal_emitter=emitter,
        clock=lambda: _dt.datetime(2026, 11, 19, 12, 0, tzinfo=_dt.UTC),
    )

    result = await launcher.launch(
        "demo",
        {"account_id": "a1"},
        auth=_auth(),
        chat_binding={
            "chat_id": "c-1",
            "source_message_id": "m-1",
            "access_token": "must-not-leak",
        },
        correlation_id="corr-1",
    )

    assert result.status == "accepted"
    assert result.signal_id == str(emitter.signal_id)
    assert result.message == "Started; results will be posted in this chat."
    [envelope] = emitter.envelopes
    assert envelope.type == "pipeline.demo.requested"
    assert envelope.payload == {"pipeline": "demo", "input": {"account_id": "a1"}}
    assert envelope.source == "internal:pipeline_tool:demo"
    assert envelope.tenant_key == "t-1"
    assert envelope.user_id == "u-1"
    assert envelope.correlation_id == "corr-1"
    assert envelope.chat_binding == {"chat_id": "c-1", "source_message_id": "m-1"}
    assert "must-not-leak" not in str(dataclasses.asdict(envelope))
    assert envelope.identity_claim == {
        "mode": "act_as_user",
        "user_id_from": "signal.user_id",
        "user_id": "u-1",
    }
    assert "secret-token" not in str(dataclasses.asdict(envelope))


async def test_launch_returns_unavailable_when_events_are_disabled(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=None,
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"
    assert emitter.envelopes == []


async def test_launch_returns_unavailable_for_missing_trigger(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(triggers=[_trigger(id="other")]),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"
    assert emitter.envelopes == []


async def test_launch_returns_unavailable_for_a_mismatched_pipeline(launcher_steps: str) -> None:
    emitter = _Emitter()
    trigger = OrchidTriggerConfig(
        id="demo-trigger",
        on=OrchidTriggerMatchConfig(signal="pipeline.demo.requested"),
        emits=OrchidTriggerEmitConfig(
            pipeline="other",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        ),
    )
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(triggers=[trigger]),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert "targets pipeline 'other'" in result.error["message"]
    assert emitter.envelopes == []


async def test_launch_returns_unavailable_for_background_mode(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(async_mode="background"),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"
    assert "signal" in result.error["message"]
    assert emitter.envelopes == []


async def test_launch_returns_unavailable_when_async_is_disabled(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(async_enabled=False),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"
    assert emitter.envelopes == []


async def test_launch_returns_unavailable_without_an_emitter(launcher_steps: str) -> None:
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter=None,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"


async def test_launch_requires_auth(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=None)

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_auth_error"
    assert emitter.envelopes == []


async def test_launch_emit_failure_is_reported(launcher_steps: str) -> None:
    emitter = _Emitter(error=RuntimeError("queue down"))
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_emit_failed"
    assert "RuntimeError" in result.error["message"]


async def test_launcher_reads_a_late_emitter_getter(launcher_steps: str) -> None:
    emitter = _Emitter()
    holder: dict[str, object] = {"emitter": None}
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter_getter=lambda: holder["emitter"],
    )

    unavailable = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())
    assert unavailable.status == "failed"

    holder["emitter"] = emitter
    accepted = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())
    assert accepted.status == "accepted"
    assert len(emitter.envelopes) == 1


async def test_launch_rejects_a_non_object_input(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", ["not", "an", "object"], auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_input_error"
    assert emitter.envelopes == []


async def test_launch_drops_a_binding_the_trigger_does_not_respect(launcher_steps: str) -> None:
    emitter = _Emitter()
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(),
        signal_emitter=emitter,
    )

    result = await launcher.launch(
        "demo",
        {"account_id": "a1"},
        auth=_auth(),
        chat_binding={"chat_id": "c-1"},
    )

    assert result.status == "accepted"
    assert result.message == "Started; results will be available in the run history."
    [envelope] = emitter.envelopes
    assert envelope.chat_binding is None


async def test_launch_rejects_a_static_trigger_input(launcher_steps: str) -> None:
    emitter = _Emitter()
    trigger = OrchidTriggerConfig(
        id="demo-trigger",
        on=OrchidTriggerMatchConfig(signal="pipeline.demo.requested"),
        emits=OrchidTriggerEmitConfig(
            pipeline="demo",
            input={"account_id": "fixed"},
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        ),
    )
    launcher = PipelineLauncher(
        registry=_registry(),
        events_config=_events(triggers=[trigger]),
        signal_emitter=emitter,
    )

    result = await launcher.launch("demo", {"account_id": "a1"}, auth=_auth())

    assert result.status == "failed"
    assert result.error["code"] == "pipeline_async_unavailable"
    assert "input_from" in result.error["message"]
    assert emitter.envelopes == []
