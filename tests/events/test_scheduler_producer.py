"""End-to-end test of :class:`SchedulerProducer`.

The producer wires APScheduler against an :class:`OrchidScheduleStore`,
fires synthetic ``cron`` signals at the dispatcher, and updates
``last_fire_at`` / ``next_fire_at`` after each fire.

We use 1-second intervals with an aggressive ``next_run_time`` hint so
the tests don't wait the full minute.  This is the Phase 2 exit demo
in code form.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from pathlib import Path

import pytest

pytest.importorskip("apscheduler")

import aiosqlite

from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.store import OrchidScheduleRecord
from orchid_ai.events.backends.sqlite import SQLiteEventStorage
from orchid_ai.events.producers.scheduler import SchedulerProducer
from orchid_ai.events.queues.sqlite import SQLiteSignalQueue
from orchid_ai.events.schedulers.apscheduler import APSchedulerBackend

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def shared_db(tmp_path: Path):
    dsn = str(tmp_path / "scheduler.db")
    conn = await aiosqlite.connect(dsn)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")

    storage = SQLiteEventStorage(conn=conn)
    await storage.init_db()
    queue = SQLiteSignalQueue(conn=conn)
    yield {"queue": queue, "storage": storage, "conn": conn, "dsn": dsn}
    await conn.close()


# ── Tests ───────────────────────────────────────────────────


async def test_producer_fires_cron_signal_end_to_end(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    # Register a 1-second interval schedule.
    record = OrchidScheduleRecord(
        schedule_id="test-cron",
        trigger_id="test-trigger",
        cron=None,
        interval_seconds=1,
        identity_claim={"mode": "service_account", "name": "digest-bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    await storage.schedules.upsert(record)

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)

    backend = APSchedulerBackend()
    producer = SchedulerProducer(schedule_store=storage.schedules, backend=backend)
    await producer.start(dispatcher)

    # Force the registered job to fire ~immediately by re-adding it
    # with an explicit next_run_time.  The producer's refresh()
    # registered a vanilla 1-second job; we wedge in a fast first
    # fire by going through the backend directly.
    backend.add_interval(
        schedule_id="test-cron",
        seconds=1,
        callback=producer._make_callback(
            schedule_id="test-cron",
            identity_claim=record.identity_claim,
            tenant_key="t-1",
        ),
        next_run_time=_dt.datetime.now(tz=_dt.UTC) + _dt.timedelta(milliseconds=50),
    )

    # Poll for the cron signal to land in the store.
    signals: list = []
    for _ in range(40):
        signals = await storage.signals.list(type="cron")
        if signals:
            break
        await asyncio.sleep(0.05)

    await producer.stop()

    assert len(signals) >= 1
    sig = signals[0]
    assert sig.type == "cron"
    assert sig.source == "scheduler:test-cron"
    assert sig.tenant_key == "t-1"
    assert sig.payload["schedule_id"] == "test-cron"
    assert sig.dedupe_key is not None
    assert sig.dedupe_key.startswith("test-cron:")
    assert sig.identity_claim is not None
    assert sig.identity_claim["mode"] == "service_account"
    assert sig.identity_claim["name"] == "digest-bot"

    # Queue must also have the row from the dispatcher's outbox.
    [leased] = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert leased.signal_id == sig.signal_id

    # ``record_fire`` was called.
    refreshed = await storage.schedules.get("test-cron")
    assert refreshed is not None
    assert refreshed.last_fire_at is not None


async def test_producer_skips_disabled_schedules(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    enabled = OrchidScheduleRecord(
        schedule_id="enabled-one",
        trigger_id="t",
        cron=None,
        interval_seconds=1,
        identity_claim={"mode": "service_account", "name": "bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    disabled = OrchidScheduleRecord(
        schedule_id="disabled-one",
        trigger_id="t",
        cron=None,
        interval_seconds=1,
        identity_claim={"mode": "service_account", "name": "bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=False,
    )
    await storage.schedules.upsert(enabled)
    await storage.schedules.upsert(disabled)

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer = SchedulerProducer(schedule_store=storage.schedules)
    await producer.start(dispatcher)

    # The disabled schedule must not be registered.
    assert producer.backend.get_next_fire("disabled-one") is None
    assert producer.backend.get_next_fire("enabled-one") is not None

    await producer.stop()


async def test_producer_dedup_protects_against_duplicate_fires(shared_db) -> None:
    """Two callbacks racing on the same wall-clock second must result
    in exactly one persisted signal — the dedupe key holds."""
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    record = OrchidScheduleRecord(
        schedule_id="dedup-test",
        trigger_id="t",
        cron=None,
        interval_seconds=60,
        identity_claim={"mode": "service_account", "name": "bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    await storage.schedules.upsert(record)

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer = SchedulerProducer(schedule_store=storage.schedules)
    await producer.start(dispatcher)

    callback = producer._make_callback(
        schedule_id="dedup-test",
        identity_claim=record.identity_claim,
        tenant_key="t-1",
    )
    # Pin clock so both fires share the same dedupe key.
    fixed = _dt.datetime(2026, 5, 6, 7, 0, 0, tzinfo=_dt.UTC)
    producer._clock = lambda: fixed

    # Re-build the callback now that the clock has been replaced;
    # ``_make_callback`` closes over ``self._clock``.
    callback = producer._make_callback(
        schedule_id="dedup-test",
        identity_claim=record.identity_claim,
        tenant_key="t-1",
    )
    await callback()
    await callback()

    await producer.stop()

    # Exactly one signal — the second was deduplicated.
    signals = await storage.signals.list(type="cron")
    assert len(signals) == 1


async def test_producer_restart_reloads_schedules(tmp_path: Path) -> None:
    """The producer reads schedules from the durable store on every
    boot — the in-memory APScheduler jobstore is intentionally
    transient."""
    dsn = str(tmp_path / "restart.db")

    # ── Lifecycle 1: write a schedule, start producer, stop. ─
    conn1 = await aiosqlite.connect(dsn)
    conn1.row_factory = aiosqlite.Row
    await conn1.execute("PRAGMA journal_mode=WAL")
    storage1 = SQLiteEventStorage(conn=conn1)
    await storage1.init_db()
    queue1 = SQLiteSignalQueue(conn=conn1)

    record = OrchidScheduleRecord(
        schedule_id="durable-1",
        trigger_id="t",
        cron=None,
        interval_seconds=1,
        identity_claim={"mode": "service_account", "name": "bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=True,
    )
    await storage1.schedules.upsert(record)

    dispatcher1 = OrchidSignalDispatcher(store=storage1.signals, queue=queue1)
    producer1 = SchedulerProducer(schedule_store=storage1.schedules)
    await producer1.start(dispatcher1)
    assert producer1.backend.get_next_fire("durable-1") is not None
    await producer1.stop()
    await conn1.close()

    # ── Lifecycle 2: reopen, start, schedule reappears. ─────
    conn2 = await aiosqlite.connect(dsn)
    conn2.row_factory = aiosqlite.Row
    await conn2.execute("PRAGMA journal_mode=WAL")
    storage2 = SQLiteEventStorage(conn=conn2)
    await storage2.init_db()
    queue2 = SQLiteSignalQueue(conn=conn2)

    dispatcher2 = OrchidSignalDispatcher(store=storage2.signals, queue=queue2)
    producer2 = SchedulerProducer(schedule_store=storage2.schedules)
    await producer2.start(dispatcher2)
    # The same schedule_id must be live again.
    assert producer2.backend.get_next_fire("durable-1") is not None
    await producer2.stop()
    await conn2.close()
