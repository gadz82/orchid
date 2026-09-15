"""Phase-3 run-visibility tests (§26).

Coverage:

- Pydantic-level rejection of every forbidden ``(identity.mode, visibility)``
  combination from §26.3.
- Registry-level rejection of the same combinations.
- Default visibility computation for each identity flavour when
  ``visibility`` is omitted.
- ``JobRun`` row is created with the correct ``visibility`` and
  ``visibility_user_id`` for each flavour.
- DB CHECK constraint rejects a manually constructed inconsistent row.
- ``OrchidAuthContext.roles`` round-trips through
  :meth:`to_storage_dict` / :meth:`from_storage_dict`.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import aiosqlite
import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    AddressedToUserIdentity,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.errors import TriggerRegistrationError
from orchid_ai.core.events.signal import Signal
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.backends.sqlite import SQLiteEventStorage
from orchid_ai.events.registry import (
    _DEFAULT_VISIBILITY,
    build_registry_from_config,
)

# ── Pydantic forbidden combos (§26.3) ───────────────────────


@pytest.mark.parametrize(
    "identity, visibility",
    [
        (ServiceAccountIdentity(name="bot"), "actor"),
        (ServiceAccountIdentity(name="bot"), "addressed"),
        (ActAsUserIdentity(user_id_from="signal.user_id"), "addressed"),
        (ActAsUserIdentity(user_id_from="signal.user_id"), "tenant"),
        (
            AddressedToUserIdentity(service_account="bot", user_id_from="signal.user_id"),
            "actor",
        ),
    ],
)
def test_pydantic_rejects_forbidden_visibility_combos(identity, visibility) -> None:
    with pytest.raises(ValidationError) as exc_info:
        OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=identity,
            visibility=visibility,
        )
    assert "visibility" in str(exc_info.value)


@pytest.mark.parametrize(
    "identity, visibility",
    [
        (ServiceAccountIdentity(name="bot"), "tenant"),
        (ServiceAccountIdentity(name="bot"), "admin"),
        (ActAsUserIdentity(user_id_from="signal.user_id"), "actor"),
        (ActAsUserIdentity(user_id_from="signal.user_id"), "admin"),
        (
            AddressedToUserIdentity(service_account="bot", user_id_from="signal.user_id"),
            "addressed",
        ),
        (
            AddressedToUserIdentity(service_account="bot", user_id_from="signal.user_id"),
            "tenant",
        ),
    ],
)
def test_pydantic_accepts_allowed_visibility_combos(identity, visibility) -> None:
    cfg = OrchidTriggerEmitConfig(
        agent="agent",
        prompt_template="hi",
        identity=identity,
        visibility=visibility,
    )
    assert cfg.visibility == visibility


# ── Registry-level rejection (defence in depth) ─────────────


def test_registry_rejects_forbidden_visibility() -> None:
    """Build the trigger via ``model_construct`` for both layers to
    bypass the Pydantic validator and ensure the registry catches
    the bad combo on its own (defence-in-depth)."""
    bad_emit = OrchidTriggerEmitConfig.model_construct(
        agent="agent",
        prompt_template="hi",
        identity=ServiceAccountIdentity(name="bot"),
        respect_chat_binding=False,
        visibility="actor",
    )
    bad_trigger = OrchidTriggerConfig.model_construct(
        id="bad",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=bad_emit,
    )
    with pytest.raises(TriggerRegistrationError) as exc_info:
        build_registry_from_config([bad_trigger], known_agents={"agent"})
    assert "visibility" in str(exc_info.value).lower()
    assert "bad" in str(exc_info.value)


# ── Default visibility computation ──────────────────────────


@pytest.mark.parametrize(
    "identity, expected",
    [
        (ServiceAccountIdentity(name="bot"), "admin"),
        (ActAsUserIdentity(user_id_from="signal.user_id"), "actor"),
        (
            AddressedToUserIdentity(service_account="bot", user_id_from="signal.user_id"),
            "addressed",
        ),
    ],
)
def test_default_visibility_matches_identity_flavour(identity, expected) -> None:
    cfg = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=identity,
        ),
    )
    registry = build_registry_from_config([cfg], known_agents={"agent"})
    trigger = registry.get("t1")
    assert trigger is not None
    assert trigger.visibility == expected
    assert _DEFAULT_VISIBILITY[identity.mode] == expected


# ── JobSpec carries visibility into the JobRun ──────────────


def _make_signal(*, user_id: str | None = "u-7") -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type="x",
        payload={},
        source="src",
        occurred_at=now,
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=now,
        user_id=user_id,
    )


def test_act_as_user_jobspec_carries_actor_and_user_id() -> None:
    cfg = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        ),
    )
    registry = build_registry_from_config([cfg], known_agents={"agent"})
    trigger = registry.get("t1")
    spec = trigger.build_job_spec(_make_signal(user_id="u-7"))
    assert spec.visibility == "actor"
    assert spec.visibility_user_id == "u-7"


def test_service_account_jobspec_carries_admin_and_no_user() -> None:
    cfg = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="digest-bot"),
        ),
    )
    registry = build_registry_from_config([cfg], known_agents={"agent"})
    trigger = registry.get("t1")
    spec = trigger.build_job_spec(_make_signal(user_id=None))
    assert spec.visibility == "admin"
    assert spec.visibility_user_id is None


def test_addressed_to_user_jobspec_carries_addressed_and_user_id() -> None:
    cfg = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=AddressedToUserIdentity(service_account="bot", user_id_from="signal.user_id"),
        ),
    )
    registry = build_registry_from_config([cfg], known_agents={"agent"})
    trigger = registry.get("t1")
    spec = trigger.build_job_spec(_make_signal(user_id="u-9"))
    assert spec.visibility == "addressed"
    assert spec.visibility_user_id == "u-9"


def test_explicit_visibility_override_propagates() -> None:
    """A trigger that explicitly opts ``service_account`` runs into
    ``tenant`` visibility (transparency override) carries that
    through to the JobSpec."""
    cfg = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="bot"),
            visibility="tenant",
        ),
    )
    registry = build_registry_from_config([cfg], known_agents={"agent"})
    trigger = registry.get("t1")
    spec = trigger.build_job_spec(_make_signal(user_id=None))
    assert spec.visibility == "tenant"
    assert spec.visibility_user_id is None


# ── DB CHECK constraint ─────────────────────────────────────


async def test_db_check_constraint_rejects_inconsistent_row() -> None:
    """SQLite enforces the table-level CHECK that
    ``visibility_user_id`` is NULL iff visibility ∈ {tenant, admin}."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    storage = SQLiteEventStorage(conn=conn)
    await storage.init_db()

    # Need a parent signal row first (FK constraint).
    sig_id = str(_uuid.uuid4())
    now_iso = _dt.datetime.now(tz=_dt.UTC).isoformat()
    await conn.execute(
        "INSERT INTO signals "
        "(signal_id, type, source, payload, tenant_key, occurred_at, persisted_at) "
        "VALUES (?, 'x', 'src', '{}', 't-1', ?, ?)",
        (sig_id, now_iso, now_iso),
    )
    await conn.commit()

    # ``tenant`` visibility with a non-NULL user_id must be rejected.
    with pytest.raises(aiosqlite.IntegrityError):
        await conn.execute(
            "INSERT INTO job_runs "
            "(run_id, trigger_id, signal_id, attempt_number, status, "
            " agent_name, parallelism_key, spec, visibility, "
            " visibility_user_id, queued_at) "
            "VALUES (?, 't', ?, 1, 'pending', 'a', 'k', '{}', "
            "        'tenant', 'u-7', ?)",
            (str(_uuid.uuid4()), sig_id, now_iso),
        )

    # And ``actor`` with NULL user_id must be rejected.
    with pytest.raises(aiosqlite.IntegrityError):
        await conn.execute(
            "INSERT INTO job_runs "
            "(run_id, trigger_id, signal_id, attempt_number, status, "
            " agent_name, parallelism_key, spec, visibility, "
            " visibility_user_id, queued_at) "
            "VALUES (?, 't', ?, 1, 'pending', 'a', 'k', '{}', "
            "        'actor', NULL, ?)",
            (str(_uuid.uuid4()), sig_id, now_iso),
        )

    # And an unknown visibility level must be rejected.
    with pytest.raises(aiosqlite.IntegrityError):
        await conn.execute(
            "INSERT INTO job_runs "
            "(run_id, trigger_id, signal_id, attempt_number, status, "
            " agent_name, parallelism_key, spec, visibility, "
            " visibility_user_id, queued_at) "
            "VALUES (?, 't', ?, 1, 'pending', 'a', 'k', '{}', "
            "        'world', NULL, ?)",
            (str(_uuid.uuid4()), sig_id, now_iso),
        )
    await conn.close()


# ── OrchidAuthContext.roles round-trip ──────────────────────


def test_auth_context_roles_default_empty() -> None:
    ctx = OrchidAuthContext(access_token="t")
    assert ctx.roles == frozenset()


def test_auth_context_roles_custom_set() -> None:
    ctx = OrchidAuthContext(access_token="t", roles={"admin", "ops"})
    assert ctx.roles == frozenset({"admin", "ops"})


def test_auth_context_roles_round_trip_through_storage_dict() -> None:
    ctx = OrchidAuthContext(
        access_token="t",
        tenant_key="t-1",
        user_id="u-7",
        roles={"admin"},
    )
    state = ctx.to_storage_dict()
    assert state["roles"] == ["admin"]

    restored = OrchidAuthContext.from_storage_dict(
        access_token="fresh",
        expires_at=0.0,
        state=state,
    )
    assert "admin" in restored.roles
    assert restored.tenant_key == "t-1"
    assert restored.user_id == "u-7"
