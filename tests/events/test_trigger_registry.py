"""Trigger registry — registration validation + JMESPath matching."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.errors import (
    MintingProbeUnsupportedError,
    OrchidIdentityNotMintableError,
    TriggerRegistrationError,
)
from orchid_ai.core.events.signal import Signal
from orchid_ai.events.registry import (
    build_registry_from_config,
    resolve_user_id_for_signal,
)


def _signal(*, type: str = "demo.event", payload: dict | None = None) -> Signal:
    return Signal(
        type=type,
        payload=payload or {},
        source="test:fixture",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime.now(tz=_dt.UTC),
        user_id="u-1",
    )


def _trigger_config(
    *,
    id: str = "t1",
    signal_type: str = "demo.event",
    when: str | None = None,
    agent: str = "notifications",
    identity=None,
) -> OrchidTriggerConfig:
    return OrchidTriggerConfig(
        id=id,
        on=OrchidTriggerMatchConfig(signal=signal_type, when=when),
        emits=OrchidTriggerEmitConfig(
            agent=agent,
            prompt_template="Hello {{tenant_key}}",
            identity=identity or ServiceAccountIdentity(name="digest-bot"),
        ),
    )


# ── Match logic ─────────────────────────────────────────────


def test_registry_matches_by_signal_type() -> None:
    reg = build_registry_from_config([_trigger_config(id="t1")], known_agents={"notifications"})
    matches = list(reg.find_matches(_signal()))
    assert len(matches) == 1
    assert matches[0].trigger_id == "t1"


def test_registry_no_match_for_other_type() -> None:
    reg = build_registry_from_config(
        [_trigger_config(id="t1", signal_type="ticket.created")],
        known_agents={"notifications"},
    )
    matches = list(reg.find_matches(_signal(type="demo.event")))
    assert matches == []


def test_registry_when_filter_jmespath() -> None:
    reg = build_registry_from_config(
        [
            _trigger_config(
                id="hot",
                signal_type="ticket.created",
                when="payload.priority == 'high'",
            )
        ],
        known_agents={"notifications"},
    )
    high = _signal(type="ticket.created", payload={"priority": "high"})
    low = _signal(type="ticket.created", payload={"priority": "low"})
    assert [t.trigger_id for t in reg.find_matches(high)] == ["hot"]
    assert list(reg.find_matches(low)) == []


def test_registry_when_filter_against_top_level_field() -> None:
    reg = build_registry_from_config(
        [_trigger_config(when="signal.user_id == 'vip'")],
        known_agents={"notifications"},
    )
    sig_vip = Signal(
        type="demo.event",
        payload={},
        source="s",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime.now(tz=_dt.UTC),
        user_id="vip",
    )
    sig_other = Signal(
        type="demo.event",
        payload={},
        source="s",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime.now(tz=_dt.UTC),
        user_id="other",
    )
    assert list(reg.find_matches(sig_vip))
    assert not list(reg.find_matches(sig_other))


# ── Registration validation ─────────────────────────────────


def test_registry_rejects_duplicate_trigger_id() -> None:
    with pytest.raises(TriggerRegistrationError, match="duplicate"):
        build_registry_from_config(
            [_trigger_config(id="dup"), _trigger_config(id="dup")],
            known_agents={"notifications"},
        )


def test_registry_rejects_unknown_agent() -> None:
    with pytest.raises(TriggerRegistrationError, match="unknown agent"):
        build_registry_from_config(
            [_trigger_config(agent="ghost-agent")],
            known_agents={"notifications"},
        )


def test_registry_rejects_unparseable_when() -> None:
    with pytest.raises(TriggerRegistrationError, match="unparseable when"):
        build_registry_from_config(
            [_trigger_config(when="this is not !!!! jmespath @@")],
            known_agents={"notifications"},
        )


# ── Mint probe (act_as_user) ────────────────────────────────


class _MintCapableResolver:
    def mint_for_user(self, tenant_key: str, user_id: str):
        # 'I support minting in general — just not for the probe user'
        if user_id == "__probe__":
            raise OrchidIdentityNotMintableError(tenant_key, user_id)
        return object()


class _MintIncapableResolver:
    def mint_for_user(self, tenant_key: str, user_id: str):
        raise MintingProbeUnsupportedError(type(self).__name__)


def test_act_as_user_passes_when_resolver_can_mint() -> None:
    cfg = _trigger_config(identity=ActAsUserIdentity(user_id_from="signal.user_id"))
    reg = build_registry_from_config(
        [cfg],
        known_agents={"notifications"},
        identity_resolver=_MintCapableResolver(),
    )
    assert reg.get(cfg.id) is not None


def test_act_as_user_rejected_when_resolver_cannot_mint() -> None:
    cfg = _trigger_config(identity=ActAsUserIdentity(user_id_from="signal.user_id"))
    with pytest.raises(TriggerRegistrationError, match="does not implement mint_for_user"):
        build_registry_from_config(
            [cfg],
            known_agents={"notifications"},
            identity_resolver=_MintIncapableResolver(),
        )


def test_service_account_does_not_invoke_mint_probe() -> None:
    """Service-account triggers must NOT call ``mint_for_user``."""

    class _Recorder:
        def __init__(self) -> None:
            self.called = False

        def mint_for_user(self, tenant_key: str, user_id: str):
            self.called = True
            return object()

    recorder = _Recorder()
    build_registry_from_config(
        [_trigger_config(identity=ServiceAccountIdentity(name="bot"))],
        known_agents={"notifications"},
        identity_resolver=recorder,
    )
    assert recorder.called is False


# ── User-id extraction ──────────────────────────────────────


def test_resolve_user_id_for_signal_via_jmespath() -> None:
    sig = _signal(payload={"user_id": "vip-1"})
    claim = {"mode": "act_as_user", "user_id_from": "payload.user_id"}
    assert resolve_user_id_for_signal(claim, signal=sig) == "vip-1"


def test_resolve_user_id_returns_none_for_service_account_claim() -> None:
    sig = _signal()
    claim = {"mode": "service_account", "name": "bot"}
    assert resolve_user_id_for_signal(claim, signal=sig) is None


def test_resolve_user_id_returns_none_when_path_missing() -> None:
    sig = _signal(payload={})
    claim = {"mode": "act_as_user", "user_id_from": "payload.absent"}
    assert resolve_user_id_for_signal(claim, signal=sig) is None


# ── Build job spec ──────────────────────────────────────────


def test_build_job_spec_renders_prompt_and_parallelism_key() -> None:
    cfg = _trigger_config(
        id="bj",
        identity=ServiceAccountIdentity(name="digest-bot"),
    )
    cfg.emits.prompt_template = "Hello {{tenant_key}}/{{signal.type}}"
    reg = build_registry_from_config([cfg], known_agents={"notifications"})
    trigger = reg.get("bj")
    assert trigger is not None
    sig = _signal()
    spec = trigger.build_job_spec(sig)
    assert spec.agent_name == "notifications"
    assert "Hello t-1/demo.event" == spec.prompt
    assert spec.identity_claim == {"mode": "service_account", "name": "digest-bot"}
    # ``per_user`` is the default; service-account claim flips key prefix.
    assert spec.parallelism_key == "sa:t-1:digest-bot"


def test_build_job_spec_user_parallelism_key_for_act_as_user() -> None:
    cfg = _trigger_config(id="aau", identity=ActAsUserIdentity(user_id_from="signal.user_id"))
    reg = build_registry_from_config(
        [cfg],
        known_agents={"notifications"},
        identity_resolver=_MintCapableResolver(),
    )
    trigger = reg.get("aau")
    assert trigger is not None
    spec = trigger.build_job_spec(_signal())
    assert spec.parallelism_key == "user:t-1:u-1"
