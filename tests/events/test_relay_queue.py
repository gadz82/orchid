"""Tests for :class:`RelayingSignalQueue` — the publish-then-mark
adapter for external buses (Kafka / SQS / Redis Streams)."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.signal import Signal
from orchid_ai.events.backends.sqlite import SQLiteEventStorage
from orchid_ai.events.queues.inmemory import InMemorySignalQueue
from orchid_ai.events.queues.relay import (
    InMemoryBusPublisher,
    RelayingSignalQueue,
)

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def storage():
    s = SQLiteEventStorage(dsn=":memory:")
    await s.init_db()
    yield s
    await s.close()


def _signal(*, source: str = "relay-test") -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type="demo.event",
        payload={"k": "v"},
        source=source,
        occurred_at=now,
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=now,
    )


# ── Tests ───────────────────────────────────────────────────


async def test_relay_queue_publishes_and_flips_status(storage) -> None:
    publisher = InMemoryBusPublisher()
    inner = InMemorySignalQueue()
    relay = RelayingSignalQueue(inner=inner, store=storage.signals, publisher=publisher)

    sig = _signal()
    await storage.signals.insert(sig)

    msg_id = await relay.enqueue(sig.signal_id)
    assert msg_id == f"relay:{sig.signal_id}"
    assert publisher.published == [sig.signal_id]

    # Final status is 'published'.
    refreshed = await storage.signals.get(sig.signal_id)
    assert refreshed is not None
    assert refreshed.relay_status == "published"


async def test_relay_queue_leaves_pending_when_publish_fails(storage) -> None:
    publisher = InMemoryBusPublisher(fail_next=1)
    inner = InMemorySignalQueue()
    relay = RelayingSignalQueue(inner=inner, store=storage.signals, publisher=publisher)

    sig = _signal()
    await storage.signals.insert(sig)

    with pytest.raises(RuntimeError):
        await relay.enqueue(sig.signal_id)

    refreshed = await storage.signals.get(sig.signal_id)
    assert refreshed is not None
    assert refreshed.relay_status == "pending_publish"
    assert publisher.published == []


async def test_relay_queue_recovers_after_publisher_recovers(storage) -> None:
    """Operator-style scenario: publisher fails once, then succeeds.
    The second enqueue uses a fresh signal id (replaying the same id
    is the recovery producer's job — out of scope for this skeleton)."""
    publisher = InMemoryBusPublisher(fail_next=1)
    inner = InMemorySignalQueue()
    relay = RelayingSignalQueue(inner=inner, store=storage.signals, publisher=publisher)

    sig_a = _signal(source="a")
    sig_b = _signal(source="b")
    await storage.signals.insert(sig_a)
    await storage.signals.insert(sig_b)

    with pytest.raises(RuntimeError):
        await relay.enqueue(sig_a.signal_id)
    # Publisher recovered; second one succeeds.
    await relay.enqueue(sig_b.signal_id)

    a = await storage.signals.get(sig_a.signal_id)
    b = await storage.signals.get(sig_b.signal_id)
    assert a is not None and a.relay_status == "pending_publish"
    assert b is not None and b.relay_status == "published"


async def test_relay_queue_consumer_methods_raise(storage) -> None:
    publisher = InMemoryBusPublisher()
    inner = InMemorySignalQueue()
    relay = RelayingSignalQueue(inner=inner, store=storage.signals, publisher=publisher)

    with pytest.raises(NotImplementedError):
        await relay.dequeue(batch_size=1, lease_seconds=30)
    with pytest.raises(NotImplementedError):
        await relay.ack("anything")
    with pytest.raises(NotImplementedError):
        await relay.nack("anything", retry_after_seconds=1)
    with pytest.raises(NotImplementedError):
        await relay.dead_letter("anything", reason="x")


async def test_relay_queue_transaction_delegates_to_inner(storage) -> None:
    publisher = InMemoryBusPublisher()
    inner = InMemorySignalQueue()
    relay = RelayingSignalQueue(inner=inner, store=storage.signals, publisher=publisher)
    async with relay.transaction() as tx:
        # In-memory inner returns a no-op transaction; just assert
        # the context manager works.
        _ = tx
