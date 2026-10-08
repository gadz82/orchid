"""End-to-end test of :class:`SchedulerProducer`.

The producer wires APScheduler against an :class:`OrchidScheduleStore`,
fires synthetic ``cron`` signals at the dispatcher, and updates
``last_fire_at`` / ``next_fire_at`` after each fire.

We use 1-second intervals with an aggressive ``next_run_time`` hint so
the tests don't wait the full minute.  The store/queue under test are
the framework's in-memory backends; durable-store parity is covered by
the storage plugin packages.
"""

from __future__ import annotations

import asyncio
import datetime as _dt

import pytest

pytest.importorskip("apscheduler")

from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.store import OrchidScheduleRecord
from orchid_ai.events.backends.inmemory import InMemoryEventStorage
from orchid_ai.events.producers.scheduler import SchedulerProducer
from orchid_ai.events.queues.inmemory import InMemorySignalQueue
from orchid_ai.events.schedulers.apscheduler import APSchedulerBackend

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def event_store():
    storage = InMemoryEventStorage()
    await storage.init_db()
    queue = InMemorySignalQueue()
    return {"queue": queue, "storage": storage}


def _record(schedule_id: str, *, enabled: bool = True) -> OrchidScheduleRecord:
    return OrchidScheduleRecord(
        schedule_id=schedule_id,
        trigger_id="t",
        cron=None,
        interval_seconds=1,
        identity_claim={"mode": "service_account", "name": "bot", "tenant_key": "t-1"},
        last_fire_at=None,
        next_fire_at=None,
        enabled=enabled,
    )


# ── Tests ───────────────────────────────────────────────────


async def test_producer_fires_cron_signal_end_to_end(event_store) -> None:
    storage: InMemoryEventStorage = event_store["storage"]
    queue: InMemorySignalQueue = event_store["queue"]

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


async def test_producer_skips_disabled_schedules(event_store) -> None:
    storage: InMemoryEventStorage = event_store["storage"]
    queue: InMemorySignalQueue = event_store["queue"]

    await storage.schedules.upsert(_record("enabled-one", enabled=True))
    await storage.schedules.upsert(_record("disabled-one", enabled=False))

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer = SchedulerProducer(schedule_store=storage.schedules)
    await producer.start(dispatcher)

    # The disabled schedule must not be registered.
    assert producer.backend.get_next_fire("disabled-one") is None
    assert producer.backend.get_next_fire("enabled-one") is not None

    await producer.stop()


async def test_producer_dedup_protects_against_duplicate_fires(event_store) -> None:
    """Two callbacks racing on the same wall-clock second must result
    in exactly one persisted signal — the dedupe key holds."""
    storage: InMemoryEventStorage = event_store["storage"]
    queue: InMemorySignalQueue = event_store["queue"]

    record = _record("dedup-test")
    await storage.schedules.upsert(record)

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer = SchedulerProducer(schedule_store=storage.schedules)
    await producer.start(dispatcher)

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


async def test_producer_restart_reloads_schedules(event_store) -> None:
    """The producer reads schedules from the store on every boot — the
    in-memory APScheduler jobstore is intentionally transient."""
    storage: InMemoryEventStorage = event_store["storage"]
    queue: InMemorySignalQueue = event_store["queue"]

    # ── Lifecycle 1: write a schedule, start producer, stop. ─
    await storage.schedules.upsert(_record("durable-1"))

    dispatcher1 = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer1 = SchedulerProducer(schedule_store=storage.schedules)
    await producer1.start(dispatcher1)
    assert producer1.backend.get_next_fire("durable-1") is not None
    await producer1.stop()

    # ── Lifecycle 2: new producer against the same store. ────
    dispatcher2 = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    producer2 = SchedulerProducer(schedule_store=storage.schedules)
    await producer2.start(dispatcher2)
    # The same schedule_id must be live again.
    assert producer2.backend.get_next_fire("durable-1") is not None
    await producer2.stop()
