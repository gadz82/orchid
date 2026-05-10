"""Round-trip tests for the four SQLite event stores.

These tests focus on persistence semantics — JSON serialisation,
dedup behaviour at the unique index, soft-delete on triggers, the
``UNIQUE (trigger_id, signal_id, attempt_number)`` idempotency
contract on jobs, and the schedule store's ``record_fire`` updates.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.errors import SignalDuplicateError
from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.core.events.signal import Signal
from orchid_ai.core.events.store import (
    OrchidScheduleRecord,
    OrchidTriggerRecord,
)
from orchid_ai.events.backends.sqlite import SQLiteEventStorage


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def storage():
    s = SQLiteEventStorage(dsn=":memory:")
    await s.init_db()
    yield s
    await s.close()


def _make_signal(
    *,
    type: str = "demo.event",
    source: str = "fixture",
    dedupe_key: str | None = None,
    payload: dict | None = None,
    identity_claim: dict | None = None,
) -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type=type,
        payload=payload or {"a": 1, "b": "two"},
        source=source,
        occurred_at=now,
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=now,
        user_id="u-1",
        correlation_id=f"corr-{_uuid.uuid4().hex[:8]}",
        dedupe_key=dedupe_key,
        identity_claim=identity_claim,
    )


def _make_run(*, attempt_number: int = 1, signal_id: _uuid.UUID | None = None) -> JobRun:
    spec = JobSpec(
        trigger_id="t1",
        signal_id=signal_id or _uuid.uuid4(),
        agent_name="notifications",
        prompt="Build the digest",
        identity_claim={"mode": "service_account", "name": "digest-bot"},
        correlation_id="corr",
        parallelism_key="sa:t-1:digest-bot",
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=attempt_number,
        status=JobStatus.PENDING,
        queued_at=_dt.datetime.now(tz=_dt.UTC),
    )


async def _seed_signal_for_run(storage, run: JobRun) -> None:
    """``job_runs.signal_id`` is FK-constrained against ``signals``,
    so every JobRun in these tests needs a backing Signal row."""
    sig = _make_signal()
    sig = Signal(
        type=sig.type,
        payload=sig.payload,
        source=sig.source,
        occurred_at=sig.occurred_at,
        tenant_key=sig.tenant_key,
        signal_id=run.spec.signal_id,
        persisted_at=sig.persisted_at,
        user_id=sig.user_id,
        correlation_id=sig.correlation_id,
        dedupe_key=None,
        identity_claim=sig.identity_claim,
    )
    await storage.signals.insert(sig)


# ── SignalStore ─────────────────────────────────────────────


async def test_signal_insert_and_get(storage) -> None:
    sig = _make_signal(payload={"nested": {"x": [1, 2, 3]}}, identity_claim={"mode": "service_account", "name": "bot"})
    await storage.signals.insert(sig)

    fetched = await storage.signals.get(sig.signal_id)
    assert fetched is not None
    assert fetched.signal_id == sig.signal_id
    assert fetched.payload == {"nested": {"x": [1, 2, 3]}}
    assert fetched.identity_claim == {"mode": "service_account", "name": "bot"}
    assert fetched.tenant_key == "t-1"


async def test_signal_dedup_within_same_source(storage) -> None:
    a = _make_signal(source="src", dedupe_key="abc")
    b = _make_signal(source="src", dedupe_key="abc")
    await storage.signals.insert(a)
    with pytest.raises(SignalDuplicateError):
        await storage.signals.insert(b)
    found = await storage.signals.find_by_dedupe(source="src", dedupe_key="abc")
    assert found == a.signal_id


async def test_signal_no_dedup_when_dedupe_key_is_none(storage) -> None:
    a = _make_signal(source="src", dedupe_key=None)
    b = _make_signal(source="src", dedupe_key=None)
    await storage.signals.insert(a)
    await storage.signals.insert(b)


async def test_signal_list_filters(storage) -> None:
    a = _make_signal(type="alpha")
    b = _make_signal(type="beta")
    await storage.signals.insert(a)
    await storage.signals.insert(b)

    only_alpha = await storage.signals.list(type="alpha")
    assert len(only_alpha) == 1
    assert only_alpha[0].type == "alpha"

    everything = await storage.signals.list()
    assert {s.type for s in everything} == {"alpha", "beta"}


# ── JobStore ────────────────────────────────────────────────


async def test_job_insert_and_get(storage) -> None:
    run = _make_run()
    await _seed_signal_for_run(storage, run)
    inserted = await storage.jobs.insert(run)
    assert inserted.run_id == run.run_id

    fetched = await storage.jobs.get(run.run_id)
    assert fetched is not None
    assert fetched.run_id == run.run_id
    assert fetched.attempt_number == 1
    assert fetched.spec.trigger_id == "t1"
    assert fetched.spec.parallelism_key == "sa:t-1:digest-bot"
    assert fetched.status == JobStatus.PENDING


async def test_job_unique_constraint_returns_existing(storage) -> None:
    """A redelivered queue message inserts the same (trigger, signal,
    attempt).  The store must return the existing run, not raise."""
    sid = _uuid.uuid4()
    a = _make_run(signal_id=sid, attempt_number=1)
    await _seed_signal_for_run(storage, a)
    b = _make_run(signal_id=sid, attempt_number=1)
    b.spec = JobSpec(
        trigger_id=a.spec.trigger_id,
        signal_id=sid,
        agent_name=a.spec.agent_name,
        prompt=a.spec.prompt,
        identity_claim=a.spec.identity_claim,
        correlation_id=a.spec.correlation_id,
        parallelism_key=a.spec.parallelism_key,
    )

    await storage.jobs.insert(a)
    returned = await storage.jobs.insert(b)
    # Same row — the existing run wins.
    assert returned.run_id == a.run_id


async def test_job_update_status_and_result(storage) -> None:
    run = _make_run()
    await _seed_signal_for_run(storage, run)
    await storage.jobs.insert(run)

    run.status = JobStatus.SUCCEEDED
    run.started_at = _dt.datetime.now(tz=_dt.UTC)
    run.finished_at = _dt.datetime.now(tz=_dt.UTC)
    run.result = {"summary": "ok", "items": [1, 2]}
    run.metadata = {"by": "tester"}
    await storage.jobs.update(run)

    fetched = await storage.jobs.get(run.run_id)
    assert fetched is not None
    assert fetched.status == JobStatus.SUCCEEDED
    assert fetched.result == {"summary": "ok", "items": [1, 2]}
    assert fetched.metadata == {"by": "tester"}
    assert fetched.started_at is not None


async def test_job_latest_attempt_and_find_latest(storage) -> None:
    sid = _uuid.uuid4()
    r1 = _make_run(signal_id=sid, attempt_number=1)
    await _seed_signal_for_run(storage, r1)
    r2 = _make_run(signal_id=sid, attempt_number=2)
    # Patch r2's spec to share trigger_id+signal_id with r1.
    r2.spec = JobSpec(
        trigger_id=r1.spec.trigger_id,
        signal_id=sid,
        agent_name=r1.spec.agent_name,
        prompt=r1.spec.prompt,
        identity_claim=r1.spec.identity_claim,
        correlation_id=r1.spec.correlation_id,
        parallelism_key=r1.spec.parallelism_key,
    )

    await storage.jobs.insert(r1)
    await storage.jobs.insert(r2)

    n = await storage.jobs.latest_attempt(trigger_id="t1", signal_id=sid)
    assert n == 2

    latest = await storage.jobs.find_latest(trigger_id="t1", signal_id=sid)
    assert latest is not None
    assert latest.run_id == r2.run_id
    assert latest.attempt_number == 2


async def test_job_list_filters(storage) -> None:
    a = _make_run()
    b = _make_run()
    await _seed_signal_for_run(storage, a)
    await _seed_signal_for_run(storage, b)
    a.status = JobStatus.SUCCEEDED
    b.status = JobStatus.FAILED
    await storage.jobs.insert(a)
    await storage.jobs.insert(b)

    succeeded = await storage.jobs.list(status="succeeded")
    assert len(succeeded) == 1
    assert succeeded[0].run_id == a.run_id


# ── ScheduleStore ───────────────────────────────────────────


async def test_schedule_upsert_and_get(storage) -> None:
    rec = OrchidScheduleRecord(
        schedule_id="morning-digest",
        trigger_id="morning-digest-trigger",
        cron="0 7 * * 1-5",
        interval_seconds=None,
        identity_claim={"mode": "service_account", "name": "digest-bot"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    await storage.schedules.upsert(rec)

    fetched = await storage.schedules.get("morning-digest")
    assert fetched is not None
    assert fetched.cron == "0 7 * * 1-5"
    assert fetched.identity_claim["name"] == "digest-bot"
    assert fetched.enabled is True


async def test_schedule_set_enabled_and_record_fire(storage) -> None:
    rec = OrchidScheduleRecord(
        schedule_id="hourly",
        trigger_id="some-trigger",
        cron=None,
        interval_seconds=3600,
        identity_claim={"mode": "service_account", "name": "bot"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    await storage.schedules.upsert(rec)

    fired_at = _dt.datetime.now(tz=_dt.UTC)
    next_at = fired_at + _dt.timedelta(hours=1)
    await storage.schedules.record_fire("hourly", last_fire_at=fired_at, next_fire_at=next_at)
    refreshed = await storage.schedules.get("hourly")
    assert refreshed is not None
    assert refreshed.last_fire_at is not None
    assert refreshed.next_fire_at is not None

    await storage.schedules.set_enabled("hourly", enabled=False)
    refreshed = await storage.schedules.get("hourly")
    assert refreshed is not None
    assert refreshed.enabled is False


async def test_schedule_list_returns_all(storage) -> None:
    rec_a = OrchidScheduleRecord(
        schedule_id="a",
        trigger_id="t",
        cron="* * * * *",
        interval_seconds=None,
        identity_claim={"mode": "service_account", "name": "bot"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    rec_b = OrchidScheduleRecord(
        schedule_id="b",
        trigger_id="t",
        cron=None,
        interval_seconds=60,
        identity_claim={"mode": "service_account", "name": "bot"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=False,
    )
    await storage.schedules.upsert(rec_a)
    await storage.schedules.upsert(rec_b)
    items = list(await storage.schedules.list())
    assert {r.schedule_id for r in items} == {"a", "b"}


# ── TriggerStore ────────────────────────────────────────────


async def test_trigger_insert_and_latest(storage) -> None:
    now = _dt.datetime.now(tz=_dt.UTC)
    v1 = OrchidTriggerRecord(
        trigger_id="t1",
        version=1,
        config={"on": {"signal": "demo"}, "emits": {"agent": "a"}},
        created_at=now,
        deleted_at=None,
    )
    v2 = OrchidTriggerRecord(
        trigger_id="t1",
        version=2,
        config={"on": {"signal": "demo"}, "emits": {"agent": "b"}},
        created_at=now + _dt.timedelta(minutes=1),
        deleted_at=None,
    )
    await storage.triggers.insert_version(v1)
    await storage.triggers.insert_version(v2)

    latest = await storage.triggers.latest("t1")
    assert latest is not None
    assert latest.version == 2
    assert latest.config["emits"]["agent"] == "b"


async def test_trigger_soft_delete_excludes_from_latest(storage) -> None:
    now = _dt.datetime.now(tz=_dt.UTC)
    rec = OrchidTriggerRecord(
        trigger_id="t1",
        version=1,
        config={"x": 1},
        created_at=now,
        deleted_at=None,
    )
    await storage.triggers.insert_version(rec)
    await storage.triggers.soft_delete("t1", deleted_at=now)
    assert await storage.triggers.latest("t1") is None


async def test_trigger_list_active_returns_only_latest(storage) -> None:
    now = _dt.datetime.now(tz=_dt.UTC)
    a1 = OrchidTriggerRecord(trigger_id="a", version=1, config={"v": 1}, created_at=now, deleted_at=None)
    a2 = OrchidTriggerRecord(trigger_id="a", version=2, config={"v": 2}, created_at=now, deleted_at=None)
    b1 = OrchidTriggerRecord(trigger_id="b", version=1, config={"v": 1}, created_at=now, deleted_at=None)
    await storage.triggers.insert_version(a1)
    await storage.triggers.insert_version(a2)
    await storage.triggers.insert_version(b1)

    active = list(await storage.triggers.list_active())
    by_id = {t.trigger_id: t for t in active}
    assert by_id["a"].version == 2
    assert by_id["b"].version == 1
