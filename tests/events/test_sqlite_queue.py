"""Behaviour of :class:`SQLiteSignalQueue` against an on-disk SQLite DB.

These tests cover the durable queue contract: visibility timers,
leases, ack/nack idempotency, dead-letter, priority ordering, and
the dispatcher's transactional outbox.  The store-side tests live in
``test_sqlite_event_store.py``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid
from pathlib import Path

import aiosqlite
import pytest

from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.signal import Signal, SignalEnvelope
from orchid_ai.events.backends.sqlite import (
    SQLiteEventStorage,
)
from orchid_ai.events.queues.sqlite import SQLiteSignalQueue

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def shared_db(tmp_path: Path):
    """Open a single connection that the queue and store both share —
    matches the production shape where they hang off the same pool."""
    dsn = str(tmp_path / "events.db")
    conn = await aiosqlite.connect(dsn)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")

    storage = SQLiteEventStorage(conn=conn)
    await storage.init_db()

    queue = SQLiteSignalQueue(conn=conn)
    yield {"queue": queue, "storage": storage, "conn": conn, "dsn": dsn}
    await conn.close()


# ── Helpers ─────────────────────────────────────────────────


def _signal(*, source: str = "test:fixture", dedupe_key: str | None = None) -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type="demo.event",
        payload={"k": "v"},
        source=source,
        occurred_at=now,
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=now,
        dedupe_key=dedupe_key,
    )


# ── Tests ───────────────────────────────────────────────────


async def test_enqueue_dequeue_ack_round_trip(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal(source="round-trip")
    await storage.signals.insert(sig)

    msg_id = await queue.enqueue(sig.signal_id)
    assert isinstance(msg_id, str)

    [leased] = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert leased.queue_msg_id == msg_id
    assert leased.signal_id == sig.signal_id
    assert leased.attempt == 1

    await queue.ack(msg_id)

    empty = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert empty == []


async def test_dequeue_respects_visibility_timer(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    await queue.nack(leased.queue_msg_id, retry_after_seconds=60)

    none_yet = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert none_yet == []


async def test_lease_expiry_makes_message_visible_again(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [first] = await queue.dequeue(batch_size=1, lease_seconds=0)
    await asyncio.sleep(0.02)

    [second] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert second.queue_msg_id == first.queue_msg_id
    assert second.attempt == 2


async def test_priority_ordering(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    low = _signal(source="low")
    high = _signal(source="high")
    await storage.signals.insert(low)
    await storage.signals.insert(high)

    await queue.enqueue(low.signal_id, priority=0)
    await queue.enqueue(high.signal_id, priority=10)

    [first] = await queue.dequeue(batch_size=1, lease_seconds=30)
    [second] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert first.signal_id == high.signal_id
    assert second.signal_id == low.signal_id


async def test_nack_dead_letters_at_max_attempts(shared_db) -> None:
    queue = SQLiteSignalQueue(conn=shared_db["conn"], max_attempts=2)
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal(source="dl-test")
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [m1] = await queue.dequeue(batch_size=1, lease_seconds=0)
    await queue.nack(m1.queue_msg_id, retry_after_seconds=0)

    await asyncio.sleep(0.02)
    [m2] = await queue.dequeue(batch_size=1, lease_seconds=0)
    assert m2.attempt == 2

    await queue.nack(m2.queue_msg_id, retry_after_seconds=0)

    assert await queue.dead_letter_count() == 1
    # Original row is gone from signal_queue.
    empty = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert empty == []


async def test_dead_letter_explicit(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal(source="dl-explicit")
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    await queue.dead_letter(leased.queue_msg_id, reason="terminal")
    assert await queue.dead_letter_count() == 1


async def test_ack_is_idempotent(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)
    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)

    await queue.ack(leased.queue_msg_id)
    # Second ack must be a no-op.
    await queue.ack(leased.queue_msg_id)


async def test_dispatcher_outbox_atomic_on_success(shared_db) -> None:
    """The dispatcher's ``ingest`` should produce both a signals row
    and a signal_queue row that share the same atomic commit."""
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)

    envelope = SignalEnvelope(
        type="demo.event",
        payload={"k": "v"},
        source="dispatcher-outbox",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
    )
    result = await dispatcher.ingest(envelope)
    assert result.deduplicated is False

    sig = await storage.signals.get(result.signal_id)
    assert sig is not None

    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert leased.signal_id == result.signal_id
    assert leased.queue_msg_id == result.queue_msg_id


async def test_dispatcher_outbox_dedup_does_not_enqueue(shared_db) -> None:
    """When the signal is a dedup hit, the dispatcher must NOT enqueue
    a second queue row — the original message handles processing."""
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)

    envelope = SignalEnvelope(
        type="demo.event",
        payload={"k": "v"},
        source="dedup-test",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        dedupe_key="abc",
    )
    first = await dispatcher.ingest(envelope)
    second = await dispatcher.ingest(envelope)

    assert first.signal_id == second.signal_id
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.queue_msg_id is None

    # Only one queue row exists for the deduplicated signal.
    [leased] = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert leased.signal_id == first.signal_id

    # Calling dequeue again should now find nothing — there was only
    # one enqueue total.
    none_yet = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert none_yet == []


async def test_queue_survives_process_restart(tmp_path: Path) -> None:
    """Phase 2 exit demo (codified): enqueue against on-disk SQLite,
    close every connection, reopen, and verify the queue still has
    the row visible."""
    dsn = str(tmp_path / "durable.db")

    # ── Lifecycle 1: write a signal + enqueue, then close. ──
    conn1 = await aiosqlite.connect(dsn)
    conn1.row_factory = aiosqlite.Row
    await conn1.execute("PRAGMA journal_mode=WAL")
    storage1 = SQLiteEventStorage(conn=conn1)
    await storage1.init_db()
    queue1 = SQLiteSignalQueue(conn=conn1)

    sig = _signal(source="restart-test", dedupe_key="restart-1")
    await storage1.signals.insert(sig)
    msg_id = await queue1.enqueue(sig.signal_id)
    await conn1.close()

    # ── Lifecycle 2: re-open everything fresh. ──────────────
    conn2 = await aiosqlite.connect(dsn)
    conn2.row_factory = aiosqlite.Row
    await conn2.execute("PRAGMA journal_mode=WAL")
    storage2 = SQLiteEventStorage(conn=conn2)
    await storage2.init_db()
    queue2 = SQLiteSignalQueue(conn=conn2)

    [leased] = await queue2.dequeue(batch_size=10, lease_seconds=30)
    assert leased.queue_msg_id == msg_id
    assert leased.signal_id == sig.signal_id

    # Signal store also persisted.
    refetched = await storage2.signals.get(sig.signal_id)
    assert refetched is not None
    assert refetched.source == "restart-test"
    assert refetched.dedupe_key == "restart-1"

    await queue2.ack(msg_id)
    await conn2.close()


async def test_queue_rejects_double_construction_modes() -> None:
    with pytest.raises(ValueError):
        SQLiteSignalQueue()
    with pytest.raises(ValueError):
        SQLiteSignalQueue(conn=None, dsn=None)


async def test_signal_store_dedup_round_trip(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    sig_a = _signal(source="src-a", dedupe_key="key")
    sig_b = _signal(source="src-b", dedupe_key="key")

    await storage.signals.insert(sig_a)
    await storage.signals.insert(sig_b)  # different source, same dedupe

    found_a = await storage.signals.find_by_dedupe(source="src-a", dedupe_key="key")
    found_b = await storage.signals.find_by_dedupe(source="src-b", dedupe_key="key")
    assert found_a == sig_a.signal_id
    assert found_b == sig_b.signal_id

    none_keyless = await storage.signals.find_by_dedupe(source="src-a", dedupe_key=None)
    assert none_keyless is None


async def test_signal_store_relay_status_round_trip(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    sig = _signal(source="relay-status")
    await storage.signals.insert(sig)
    await storage.signals.update_relay_status(sig.signal_id, status="published")
    refreshed = await storage.signals.get(sig.signal_id)
    assert refreshed is not None
    assert refreshed.relay_status == "published"


async def test_visible_in_flight_dlq_counts(shared_db) -> None:
    queue: SQLiteSignalQueue = shared_db["queue"]
    storage: SQLiteEventStorage = shared_db["storage"]

    a = _signal(source="counts-a")
    b = _signal(source="counts-b")
    await storage.signals.insert(a)
    await storage.signals.insert(b)

    await queue.enqueue(a.signal_id)
    await queue.enqueue(b.signal_id)
    assert await queue.visible_count() == 2

    [_leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert await queue.visible_count() == 1
    assert await queue.in_flight_count() == 1

    # Second message dead-letter explicitly.
    [other] = await queue.dequeue(batch_size=1, lease_seconds=30)
    await queue.dead_letter(other.queue_msg_id, reason="forced")
    assert await queue.dead_letter_count() == 1
