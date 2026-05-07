"""Round-trip tests for the immutable signal value objects."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.signal import (
    Signal,
    SignalEnvelope,
    SignalIngestResult,
)


def test_envelope_is_frozen_and_slotted() -> None:
    env = SignalEnvelope(
        type="demo.event",
        payload={"a": 1},
        source="test",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
    )
    # ``frozen=True`` blocks both reassignment and new attribute creation.
    # CPython raises FrozenInstanceError (subclass of AttributeError);
    # the slots+frozen combo also surfaces TypeError from
    # ``object.__setattr__`` on some interpreters — accept either.
    with pytest.raises((AttributeError, TypeError)):
        env.type = "mutated"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        env.surprise = "boom"  # type: ignore[attr-defined]


def test_signal_from_envelope_round_trip() -> None:
    env = SignalEnvelope(
        type="demo.event",
        payload={"priority": "high"},
        source="test:fixture",
        occurred_at=_dt.datetime(2025, 1, 1, tzinfo=_dt.UTC),
        tenant_key="t-1",
        user_id="u-1",
        correlation_id="corr-7",
        dedupe_key="abc-123",
        identity_claim={"mode": "service_account", "name": "digest-bot"},
    )
    sid = _uuid.uuid4()
    persisted = _dt.datetime(2025, 1, 2, tzinfo=_dt.UTC)
    sig = Signal.from_envelope(env, signal_id=sid, persisted_at=persisted)

    assert sig.signal_id == sid
    assert sig.persisted_at == persisted
    assert sig.type == env.type
    assert sig.payload == env.payload
    assert sig.dedupe_key == "abc-123"
    assert sig.identity_claim == {"mode": "service_account", "name": "digest-bot"}
    # ``relay_status`` defaults to "committed"
    assert sig.relay_status == "committed"


def test_identity_claim_is_copied_not_aliased() -> None:
    claim = {"mode": "act_as_user", "user_id_from": "signal.user_id"}
    env = SignalEnvelope(
        type="x",
        payload={},
        source="s",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t",
        identity_claim=claim,
    )
    sig = Signal.from_envelope(env, signal_id=_uuid.uuid4(), persisted_at=_dt.datetime.now(tz=_dt.UTC))
    claim["mode"] = "mutated"
    assert sig.identity_claim == {
        "mode": "act_as_user",
        "user_id_from": "signal.user_id",
    }


def test_as_match_dict_shape() -> None:
    sig = Signal(
        type="ticket.created",
        payload={"priority": "high", "summary": "down"},
        source="acme:webhook",
        occurred_at=_dt.datetime(2025, 6, 1, tzinfo=_dt.UTC),
        tenant_key="t-9",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime(2025, 6, 1, tzinfo=_dt.UTC),
        user_id="u-42",
        correlation_id="cx",
    )
    m = sig.as_match_dict()
    assert m["signal"]["type"] == "ticket.created"
    assert m["signal"]["user_id"] == "u-42"
    assert m["payload"] == {"priority": "high", "summary": "down"}
    # Top-level shortcuts so ``payload.priority`` JMESPath still works.
    assert m["type"] == "ticket.created"
    assert m["source"] == "acme:webhook"
    assert m["tenant_key"] == "t-9"


def test_signal_ingest_result_default_dedup_flag() -> None:
    r = SignalIngestResult(signal_id=_uuid.uuid4(), queue_msg_id="q-001")
    assert r.deduplicated is False
