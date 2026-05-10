"""Behaviour of the in-memory queue + signal store."""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.errors import SignalDuplicateError
from orchid_ai.core.events.signal import Signal
from orchid_ai.events.queues.inmemory import (
    InMemorySignalQueue,
    InMemorySignalStore,
)


def _signal(*, source: str, dedupe_key: str | None = None) -> Signal:
    return Signal(
        type="demo.event",
        payload={"k": "v"},
        source=source,
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=_dt.datetime.now(tz=_dt.UTC),
        dedupe_key=dedupe_key,
    )


# ── Store ────────────────────────────────────────────────────


async def test_store_insert_and_get_round_trip() -> None:
    store = InMemorySignalStore()
    sig = _signal(source="s")
    inserted = await store.insert(sig)
    assert inserted is sig
    fetched = await store.get(sig.signal_id)
    assert fetched is sig


async def test_store_dedup_returns_existing_id() -> None:
    store = InMemorySignalStore()
    first = _signal(source="s", dedupe_key="abc")
    await store.insert(first)
    second = _signal(source="s", dedupe_key="abc")
    with pytest.raises(SignalDuplicateError):
        await store.insert(second)
    found = await store.find_by_dedupe(source="s", dedupe_key="abc")
    assert found == first.signal_id


async def test_store_dedup_only_within_same_source() -> None:
    store = InMemorySignalStore()
    a = _signal(source="src-a", dedupe_key="key")
    b = _signal(source="src-b", dedupe_key="key")
    await store.insert(a)
    await store.insert(b)  # different source, same dedupe_key — fine
    assert await store.find_by_dedupe(source="src-a", dedupe_key="key") == a.signal_id
    assert await store.find_by_dedupe(source="src-b", dedupe_key="key") == b.signal_id


async def test_store_no_dedup_when_key_is_none() -> None:
    store = InMemorySignalStore()
    a = _signal(source="s")  # dedupe_key=None
    b = _signal(source="s")
    await store.insert(a)
    await store.insert(b)  # both succeed — None acts as 'no dedup'
    assert await store.find_by_dedupe(source="s", dedupe_key=None) is None


async def test_store_relay_status_update() -> None:
    store = InMemorySignalStore()
    sig = _signal(source="s")
    await store.insert(sig)
    await store.update_relay_status(sig.signal_id, status="published")
    refreshed = await store.get(sig.signal_id)
    assert refreshed is not None
    assert refreshed.relay_status == "published"


# ── Queue ────────────────────────────────────────────────────


async def test_queue_enqueue_dequeue_ack() -> None:
    queue = InMemorySignalQueue()
    sid = _uuid.uuid4()
    msg_id = await queue.enqueue(sid)
    assert isinstance(msg_id, str)
    assert queue.visible_messages == 1
    batch = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert len(batch) == 1
    assert batch[0].signal_id == sid
    assert batch[0].attempt == 1
    assert queue.in_flight == 1
    assert queue.visible_messages == 0
    await queue.ack(msg_id)
    assert queue.in_flight == 0


async def test_queue_dequeue_respects_batch_size_and_lease() -> None:
    queue = InMemorySignalQueue()
    ids = [_uuid.uuid4() for _ in range(5)]
    for sid in ids:
        await queue.enqueue(sid)
    first = await queue.dequeue(batch_size=2, lease_seconds=30)
    assert len(first) == 2
    second = await queue.dequeue(batch_size=10, lease_seconds=30)
    # Two are leased — only the remaining three should be visible.
    assert len(second) == 3


async def test_queue_dequeue_returns_only_visible_after_visible_after() -> None:
    queue = InMemorySignalQueue()
    await queue.enqueue(_uuid.uuid4())
    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    # Nack with a positive backoff — message becomes visible later.
    await queue.nack(leased.queue_msg_id, retry_after_seconds=60)
    none_yet = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert none_yet == []


async def test_queue_lease_expiry_makes_message_visible_again() -> None:
    queue = InMemorySignalQueue()
    await queue.enqueue(_uuid.uuid4())
    # Tiny lease so it expires within the test.
    [first] = await queue.dequeue(batch_size=1, lease_seconds=0)
    # Lease elapsed instantly.  Wait one event-loop tick so wall clock
    # advances past the lease boundary.
    await asyncio.sleep(0.01)
    [second] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert second.queue_msg_id == first.queue_msg_id
    assert second.attempt == 2
    await queue.ack(first.queue_msg_id)


async def test_queue_nack_dead_letters_at_max_attempts() -> None:
    queue = InMemorySignalQueue(max_attempts=2)
    await queue.enqueue(_uuid.uuid4())
    [m1] = await queue.dequeue(batch_size=1, lease_seconds=0)
    await queue.nack(m1.queue_msg_id, retry_after_seconds=0)
    await asyncio.sleep(0.01)
    [m2] = await queue.dequeue(batch_size=1, lease_seconds=0)
    # Second attempt — nacking again should dead-letter, not nack.
    await queue.nack(m2.queue_msg_id, retry_after_seconds=0)
    assert queue.dead_letters
    [dl] = list(queue.dead_letters.values())
    assert dl.queue_msg_id == m1.queue_msg_id
    assert dl.attempts >= 2


async def test_queue_dead_letter_explicit() -> None:
    queue = InMemorySignalQueue()
    await queue.enqueue(_uuid.uuid4())
    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    await queue.dead_letter(leased.queue_msg_id, reason="terminal failure")
    assert queue.dead_letters[leased.queue_msg_id].reason == "terminal failure"


async def test_queue_ack_idempotent() -> None:
    queue = InMemorySignalQueue()
    await queue.enqueue(_uuid.uuid4())
    [leased] = await queue.dequeue(batch_size=1, lease_seconds=30)
    await queue.ack(leased.queue_msg_id)
    # Acking again must not raise.
    await queue.ack(leased.queue_msg_id)


async def test_queue_priority_ordering() -> None:
    queue = InMemorySignalQueue()
    low = _uuid.uuid4()
    high = _uuid.uuid4()
    await queue.enqueue(low, priority=0)
    await queue.enqueue(high, priority=10)
    [first] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert first.signal_id == high
    [second] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert second.signal_id == low
