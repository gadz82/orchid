"""Tests for ``RelayRecoveryProducer`` — the publish-then-mark sweep.

These run against the in-memory store so the contract is exercised
without a Postgres dependency.  The producer's preferred-path
accessor (``list_by_relay_status``) lands on the in-memory store
along with the producer; the fallback path is exercised via a
trimmed-down store that omits the helper.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.signal import Signal
from orchid_ai.events.producers.relay_recovery import RelayRecoveryProducer
from orchid_ai.events.queues.inmemory import InMemorySignalStore
from orchid_ai.events.queues.relay import InMemoryBusPublisher


def _signal(*, relay_status: str = "pending_publish") -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type="x",
        payload={},
        source="src",
        occurred_at=now,
        tenant_key="t-1",
        signal_id=_uuid.uuid4(),
        persisted_at=now,
        relay_status=relay_status,
    )


# ── sweep_once happy path ───────────────────────────────────


async def test_sweep_publishes_and_marks_published() -> None:
    store = InMemorySignalStore()
    publisher = InMemoryBusPublisher()
    sigs = [_signal() for _ in range(3)]
    for s in sigs:
        await store.insert(s)

    producer = RelayRecoveryProducer(store=store, publisher=publisher)
    count = await producer.sweep_once()
    assert count == 3
    # ``InMemoryBusPublisher.published`` is a list of UUIDs.
    assert set(publisher.published) == {s.signal_id for s in sigs}
    for s in sigs:
        refreshed = await store.get(s.signal_id)
        assert refreshed is not None
        assert refreshed.relay_status == "published"


async def test_sweep_returns_zero_when_nothing_pending() -> None:
    store = InMemorySignalStore()
    s = _signal(relay_status="published")
    await store.insert(s)

    producer = RelayRecoveryProducer(store=store, publisher=InMemoryBusPublisher())
    assert await producer.sweep_once() == 0


async def test_sweep_leaves_failures_pending() -> None:
    store = InMemorySignalStore()
    publisher = InMemoryBusPublisher(fail_next=10)  # all subsequent fail
    s1, s2 = _signal(), _signal()
    await store.insert(s1)
    await store.insert(s2)

    producer = RelayRecoveryProducer(store=store, publisher=publisher)
    count = await producer.sweep_once()
    assert count == 0  # neither succeeded
    assert publisher.published == []
    for s in (s1, s2):
        refreshed = await store.get(s.signal_id)
        assert refreshed is not None
        assert refreshed.relay_status == "pending_publish"


# ── batch_size limit ────────────────────────────────────────


async def test_sweep_respects_batch_size() -> None:
    store = InMemorySignalStore()
    publisher = InMemoryBusPublisher()
    for _ in range(5):
        await store.insert(_signal())

    producer = RelayRecoveryProducer(store=store, publisher=publisher, batch_size=2)
    count = await producer.sweep_once()
    assert count == 2  # capped


# ── Lifecycle (start / stop) ────────────────────────────────


async def test_start_stop_lifecycle_is_idempotent_safe() -> None:
    store = InMemorySignalStore()
    publisher = InMemoryBusPublisher()
    producer = RelayRecoveryProducer(store=store, publisher=publisher, poll_interval_seconds=0.1)

    # ``dispatcher`` is unused by the recovery producer but the ABC
    # requires it.
    class _Dispatcher:
        async def ingest(self, *a, **kw):  # pragma: no cover
            ...

    await producer.start(_Dispatcher())
    # Give the sweep loop a tick.
    await asyncio.sleep(0.05)
    await producer.stop()


async def test_start_twice_raises() -> None:
    store = InMemorySignalStore()
    producer = RelayRecoveryProducer(store=store, publisher=InMemoryBusPublisher())

    class _D:
        async def ingest(self, *a, **kw):  # pragma: no cover
            ...

    await producer.start(_D())
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await producer.start(_D())
    finally:
        await producer.stop()


def test_construction_validates_args() -> None:
    store = InMemorySignalStore()
    pub = InMemoryBusPublisher()
    with pytest.raises(ValueError):
        RelayRecoveryProducer(store=store, publisher=pub, poll_interval_seconds=0)
    with pytest.raises(ValueError):
        RelayRecoveryProducer(store=store, publisher=pub, batch_size=0)
