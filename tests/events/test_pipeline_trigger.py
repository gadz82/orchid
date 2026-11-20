"""Pipeline targets on Bloom triggers: schema, boot validation and job specs."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidEventsConfig,
    OrchidProcessorConfig,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.errors import TriggerMatchError, TriggerRegistrationError
from orchid_ai.core.events.signal import Signal
from orchid_ai.events.bootstrap import start_events, stop_events
from orchid_ai.events.registry import build_registry_from_config


def _signal(*, payload: dict | None = None, user_id: str = "u-1") -> Signal:
    return Signal(
        type="pipeline.requested",
        payload=payload or {},
        source="test:fixture",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime.now(tz=_dt.UTC),
        user_id=user_id,
    )


def _pipeline_emit(**overrides) -> OrchidTriggerEmitConfig:
    values = {
        "pipeline": "weekly_digest",
        "identity": ServiceAccountIdentity(name="digest-bot"),
    }
    values.update(overrides)
    return OrchidTriggerEmitConfig(**values)


def _pipeline_trigger(**emit_overrides) -> OrchidTriggerConfig:
    return OrchidTriggerConfig(
        id="pipe",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=_pipeline_emit(**emit_overrides),
    )


# ── Schema ──────────────────────────────────────────────────


def test_pipeline_trigger_parses_without_agent_or_prompt() -> None:
    cfg = _pipeline_trigger(input={"topic": "{{payload.topic}}"})
    assert cfg.emits.agent == ""
    assert cfg.emits.prompt_template == ""
    assert cfg.emits.pipeline == "weekly_digest"
    assert cfg.emits.input == {"topic": "{{payload.topic}}"}


def test_trigger_requires_exactly_one_target() -> None:
    with pytest.raises(ValidationError, match="exactly one of 'agent' or 'pipeline'"):
        OrchidTriggerEmitConfig(
            identity=ServiceAccountIdentity(name="bot"),
        )
    with pytest.raises(ValidationError, match="exactly one of 'agent' or 'pipeline'"):
        OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hello",
            pipeline="weekly_digest",
            identity=ServiceAccountIdentity(name="bot"),
        )


def test_pipeline_trigger_rejects_prompt_template() -> None:
    with pytest.raises(ValidationError, match="must not declare 'prompt_template'"):
        _pipeline_emit(prompt_template="hello")


def test_agent_trigger_ignores_pipeline_fields() -> None:
    cfg = OrchidTriggerEmitConfig(
        agent="notifications",
        prompt_template="hello",
        input={"topic": "x"},
        input_from="payload",
        persist_graph=True,
        identity=ServiceAccountIdentity(name="bot"),
    )
    assert cfg.agent == "notifications"
    assert cfg.persist_graph is True  # parsed but ignored by the agent target


def test_pipeline_trigger_rejects_invalid_name() -> None:
    with pytest.raises(ValidationError, match="not a valid pipeline name"):
        _pipeline_emit(pipeline="Weekly Digest")


def test_pipeline_trigger_accepts_input_overlay_on_input_from() -> None:
    cfg = _pipeline_emit(input={"limit": 3}, input_from="payload")
    assert cfg.input_from == "payload"
    assert cfg.input == {"limit": 3}


def test_pipeline_trigger_rejects_bad_input_from_expression() -> None:
    with pytest.raises(ValidationError):
        _pipeline_emit(input_from="payload[")


def test_pipeline_trigger_rejects_bad_input_template() -> None:
    with pytest.raises(ValidationError):
        _pipeline_emit(input={"topic": "{{payload.topic"})
    with pytest.raises(ValidationError, match="unbalanced"):
        _pipeline_emit(input={"nested": {"topic": "{{bad["}})
    with pytest.raises(ValidationError, match="unbalanced"):
        _pipeline_emit(input={"topic": "a }} b"})
    with pytest.raises(ValidationError, match="unbalanced"):
        _pipeline_emit(input={"items": ["{{bad["]})


def test_pipeline_trigger_rejects_nul_input() -> None:
    with pytest.raises(ValidationError, match="NUL"):
        _pipeline_emit(input={"topic": "a\x00b"})
    with pytest.raises(ValidationError, match="NUL"):
        _pipeline_emit(input={"a\x00b": 1})


def test_pipeline_trigger_rejects_non_json_input() -> None:
    with pytest.raises(ValidationError, match="must be JSON-serializable"):
        _pipeline_emit(input={"since": _dt.date(2026, 11, 1)})
    with pytest.raises(ValidationError, match="must be JSON-serializable"):
        _pipeline_emit(input={"nested": {"when": _dt.datetime(2026, 11, 1, tzinfo=_dt.UTC)}})


# ── Registry boot validation ────────────────────────────────


def test_registry_rejects_unknown_pipeline() -> None:
    with pytest.raises(TriggerRegistrationError, match="unknown pipeline 'weekly_digest'"):
        build_registry_from_config(
            [_pipeline_trigger()],
            known_agents=set(),
            known_pipelines={"other_pipeline"},
        )


def test_registry_accepts_known_pipeline_without_agent() -> None:
    reg = build_registry_from_config(
        [_pipeline_trigger()],
        known_agents=set(),
        known_pipelines={"weekly_digest"},
    )
    assert reg.get("pipe") is not None


def test_agent_trigger_still_requires_known_agent() -> None:
    cfg = OrchidTriggerConfig(
        id="agent",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )
    with pytest.raises(TriggerRegistrationError, match="unknown agent"):
        build_registry_from_config([cfg], known_agents=set(), known_pipelines={"weekly_digest"})


# ── Job spec ────────────────────────────────────────────────


def _build_spec(cfg: OrchidTriggerConfig, *, signal: Signal | None = None):
    reg = build_registry_from_config(
        [cfg],
        known_agents={cfg.emits.agent} if cfg.emits.agent else set(),
        known_pipelines={"weekly_digest"},
    )
    trigger = reg.get(cfg.id)
    assert trigger is not None
    return trigger.build_job_spec(signal or _signal())


def test_pipeline_job_spec_carries_target_name_and_rendered_input() -> None:
    spec = _build_spec(
        _pipeline_trigger(input={"topic": "{{payload.topic}}", "limit": "{{payload.limit}}"}),
        signal=_signal(payload={"topic": "release notes", "limit": 3}),
    )
    assert spec.target == "pipeline"
    assert spec.pipeline_name == "weekly_digest"
    assert spec.pipeline_input == {"topic": "release notes", "limit": 3}
    assert spec.agent_name == ""
    assert spec.prompt == ""
    assert spec.persist_graph is False


def test_pipeline_job_spec_input_from_selects_signal_payload() -> None:
    spec = _build_spec(
        _pipeline_trigger(input_from="payload"),
        signal=_signal(payload={"topic": "release notes", "limit": 3}),
    )
    assert spec.pipeline_input == {"topic": "release notes", "limit": 3}


def test_pipeline_job_spec_rejects_non_json_runtime_input() -> None:
    for payload in (
        {"score": float("nan")},
        {"score": float("inf")},
        {"topic": "a\x00b"},
    ):
        reg = build_registry_from_config(
            [_pipeline_trigger(input_from="payload")],
            known_agents=set(),
            known_pipelines={"weekly_digest"},
        )
        trigger = reg.get("pipe")
        assert trigger is not None
        with pytest.raises(TriggerMatchError, match="not JSON-serializable"):
            trigger.build_job_spec(_signal(payload=payload))


def test_pipeline_job_spec_input_from_must_resolve_to_object() -> None:
    reg = build_registry_from_config(
        [_pipeline_trigger(input_from="payload.topic")],
        known_agents=set(),
        known_pipelines={"weekly_digest"},
    )
    trigger = reg.get("pipe")
    assert trigger is not None
    with pytest.raises(TriggerMatchError, match="must resolve to an object"):
        trigger.build_job_spec(_signal(payload={"topic": "release notes"}))


def test_pipeline_job_spec_applies_persist_graph() -> None:
    spec = _build_spec(_pipeline_trigger(persist_graph=True))
    assert spec.persist_graph is True


def test_pipeline_job_spec_input_overlays_input_from_base() -> None:
    spec = _build_spec(
        _pipeline_trigger(
            input_from="payload.base",
            input={"limit": 5, "topic": "{{payload.topic}}"},
        ),
        signal=_signal(
            payload={
                "base": {"topic": "base", "limit": 1, "keep": True},
                "topic": "override",
            }
        ),
    )
    assert spec.pipeline_input == {"topic": "override", "limit": 5, "keep": True}


def test_pipeline_job_spec_keeps_identity_claim_without_bearer_token() -> None:
    spec = _build_spec(
        _pipeline_trigger(identity=ActAsUserIdentity(user_id_from="signal.user_id")),
    )
    assert spec.identity_claim == {"mode": "act_as_user", "user_id_from": "signal.user_id"}
    assert "access_token" not in spec.identity_claim
    assert "token" not in {key.lower() for key in spec.identity_claim}


def test_agent_job_spec_still_defaults_to_agent_target() -> None:
    cfg = OrchidTriggerConfig(
        id="agent",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="Hello {{tenant_key}}",
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )
    reg = build_registry_from_config([cfg], known_agents={"notifications"})
    trigger = reg.get("agent")
    assert trigger is not None
    spec = trigger.build_job_spec(_signal())
    assert spec.target == "agent"
    assert spec.pipeline_name is None
    assert spec.pipeline_input is None
    assert spec.persist_graph is False
    assert spec.prompt == "Hello t-1"


def test_agent_job_spec_ignores_pipeline_only_flags() -> None:
    cfg = OrchidTriggerConfig(
        id="agent-ignore",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="Hello",
            input={"ignored": True},
            persist_graph=True,
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )
    spec = _build_spec(cfg)
    assert spec.target == "agent"
    assert spec.pipeline_name is None
    assert spec.pipeline_input is None
    assert spec.persist_graph is False


class _StubPipelineRunner:
    async def run(self, name, input, **kwargs) -> dict:
        return {"status": "succeeded", "pipeline": name}


def _events_config() -> OrchidEventsConfig:
    return OrchidEventsConfig(
        enabled=True,
        processors=[
            OrchidProcessorConfig(
                class_path="orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor",
                poll_interval_ms=10_000,
            )
        ],
        triggers=[_pipeline_trigger()],
    )


async def test_start_events_threads_known_pipelines_into_the_registry() -> None:
    runtime = await start_events(
        events_config=_events_config(),
        chat_storage=None,
        identity_resolver=None,
        session_warmer=None,
        known_agents=set(),
        known_pipelines={"weekly_digest"},
        pipeline_runner=_StubPipelineRunner(),
    )
    try:
        assert runtime.trigger_registry is not None
        assert runtime.trigger_registry.get("pipe") is not None
    finally:
        await stop_events(runtime)


async def test_start_events_rejects_pipeline_trigger_without_runner() -> None:
    with pytest.raises(TriggerRegistrationError, match="no pipeline_runner"):
        await start_events(
            events_config=_events_config(),
            chat_storage=None,
            identity_resolver=None,
            session_warmer=None,
            known_agents=set(),
            known_pipelines={"weekly_digest"},
        )


def test_agent_trigger_with_blank_pipeline_is_still_agent() -> None:
    cfg = OrchidTriggerConfig(
        id="agent-blank",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="Hello",
            pipeline="   ",
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )
    reg = build_registry_from_config([cfg], known_agents={"notifications"})
    trigger = reg.get("agent-blank")
    assert trigger is not None
    spec = trigger.build_job_spec(_signal())
    assert spec.target == "agent"
    assert spec.pipeline_name is None


async def test_start_events_ignores_blank_pipeline_on_agent_trigger() -> None:
    cfg = OrchidTriggerConfig(
        id="agent-blank",
        on=OrchidTriggerMatchConfig(signal="pipeline.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="Hello",
            pipeline="   ",
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )
    events_config = OrchidEventsConfig(
        enabled=True,
        processors=[
            OrchidProcessorConfig(
                class_path="orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor",
                poll_interval_ms=10_000,
            )
        ],
        triggers=[cfg],
    )
    runtime = await start_events(
        events_config=events_config,
        chat_storage=None,
        identity_resolver=None,
        session_warmer=None,
        known_agents={"notifications"},
    )
    try:
        assert runtime.trigger_registry is not None
        assert runtime.trigger_registry.get("agent-blank") is not None
    finally:
        await stop_events(runtime)
