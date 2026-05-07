"""Postgres-backed queue + event-store tests.

These tests require a live Postgres reachable at the DSN supplied via
``ORCHID_TEST_PG_DSN``.  They're skipped when the variable is unset
so CI can stay hermetic; the docker-compose Postgres is what wires
them up locally::

    export ORCHID_TEST_PG_DSN=postgresql://orchid:orchid@localhost:5432/orchid_test
    cd orchid && .venv/bin/python -m pytest tests/events/test_postgres_queue.py

The fixture isolates each test in a fresh schema and rolls back at
teardown so tests don't leak rows across runs.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import uuid as _uuid

import pytest

from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.signal import Signal, SignalEnvelope
from orchid_ai.events.backends.postgres import PostgresEventStorage
from orchid_ai.events.queues.postgres import PostgresSignalQueue

_PG_DSN = os.environ.get("ORCHID_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    _PG_DSN is None,
    reason="ORCHID_TEST_PG_DSN not set — Postgres tests are opt-in.",
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def shared_pg():
    """Open a private schema, run migrations, hand out the storage +
    queue, drop the schema at teardown."""
    import asyncpg

    schema = f"orchid_evt_{_uuid.uuid4().hex[:10]}"
    bootstrap = await asyncpg.connect(_PG_DSN)
    await bootstrap.execute(f'CREATE SCHEMA "{schema}"')
    await bootstrap.close()

    async def _set_schema(conn: "asyncpg.Connection") -> None:
        await conn.execute(f'SET search_path TO "{schema}"')

    pool = await asyncpg.create_pool(_PG_DSN, min_size=1, max_size=4, init=_set_schema)

    storage = PostgresEventStorage(pool=pool)
    await storage.init_db()
    queue = PostgresSignalQueue(pool=pool, notify_enabled=False)
    yield {"storage": storage, "queue": queue, "pool": pool, "schema": schema}

    await pool.close()
    bootstrap = await asyncpg.connect(_PG_DSN)
    await bootstrap.execute(f'DROP SCHEMA "{schema}" CASCADE')
    await bootstrap.close()


# ── Helpers ─────────────────────────────────────────────────


def _signal(*, source: str = "pg-test", dedupe_key: str | None = None) -> Signal:
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


async def test_pg_enqueue_dequeue_ack(shared_pg) -> None:
    queue: PostgresSignalQueue = shared_pg["queue"]
    storage: PostgresEventStorage = shared_pg["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    msg_id = await queue.enqueue(sig.signal_id)

    [leased] = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert str(leased.queue_msg_id) == msg_id
    assert leased.signal_id == sig.signal_id
    assert leased.attempt == 1

    await queue.ack(msg_id)
    empty = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert empty == []


async def test_pg_dispatcher_outbox(shared_pg) -> None:
    queue: PostgresSignalQueue = shared_pg["queue"]
    storage: PostgresEventStorage = shared_pg["storage"]

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    envelope = SignalEnvelope(
        type="demo.event",
        payload={"k": "v"},
        source="outbox",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        dedupe_key="abc",
    )
    first = await dispatcher.ingest(envelope)
    assert first.deduplicated is False

    second = await dispatcher.ingest(envelope)
    assert second.deduplicated is True
    assert second.signal_id == first.signal_id

    [leased] = await queue.dequeue(batch_size=10, lease_seconds=30)
    assert leased.signal_id == first.signal_id


async def test_pg_for_update_skip_locked_two_workers(shared_pg) -> None:
    """Two concurrent dequeue calls must each see a different message
    — the spec calls this out as the production-shape contract."""
    queue: PostgresSignalQueue = shared_pg["queue"]
    storage: PostgresEventStorage = shared_pg["storage"]

    sigs = [_signal(source=f"s-{i}") for i in range(2)]
    for sig in sigs:
        await storage.signals.insert(sig)
        await queue.enqueue(sig.signal_id)

    # Race two dequeues; SKIP LOCKED must hand each one a distinct row.
    a, b = await asyncio.gather(
        queue.dequeue(batch_size=1, lease_seconds=30),
        queue.dequeue(batch_size=1, lease_seconds=30),
    )
    ids = {a[0].queue_msg_id, b[0].queue_msg_id}
    assert len(ids) == 2  # different messages


async def test_pg_lease_expiry_redelivers(shared_pg) -> None:
    queue: PostgresSignalQueue = shared_pg["queue"]
    storage: PostgresEventStorage = shared_pg["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [first] = await queue.dequeue(batch_size=1, lease_seconds=0)
    await asyncio.sleep(1.1)
    [second] = await queue.dequeue(batch_size=1, lease_seconds=30)
    assert second.queue_msg_id == first.queue_msg_id
    assert second.attempt == 2


async def test_pg_nack_dead_letters_at_max_attempts(shared_pg) -> None:
    queue = PostgresSignalQueue(pool=shared_pg["pool"], notify_enabled=False, max_attempts=2)
    storage: PostgresEventStorage = shared_pg["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    await queue.enqueue(sig.signal_id)

    [m1] = await queue.dequeue(batch_size=1, lease_seconds=0)
    await queue.nack(m1.queue_msg_id, retry_after_seconds=0)

    await asyncio.sleep(0.05)
    [m2] = await queue.dequeue(batch_size=1, lease_seconds=0)
    assert m2.attempt == 2
    await queue.nack(m2.queue_msg_id, retry_after_seconds=0)

    assert await queue.dead_letter_count() == 1


async def test_pg_signal_store_round_trip(shared_pg) -> None:
    storage: PostgresEventStorage = shared_pg["storage"]

    sig = _signal()
    await storage.signals.insert(sig)
    fetched = await storage.signals.get(sig.signal_id)
    assert fetched is not None
    assert fetched.source == sig.source
    assert fetched.payload == {"k": "v"}
    assert fetched.tenant_key == "t-1"

    listed = await storage.signals.list()
    assert any(s.signal_id == sig.signal_id for s in listed)


async def test_pg_priority_ordering(shared_pg) -> None:
    queue: PostgresSignalQueue = shared_pg["queue"]
    storage: PostgresEventStorage = shared_pg["storage"]

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


async def test_pg_construction_modes_validated() -> None:
    with pytest.raises(ValueError):
        PostgresSignalQueue()
    with pytest.raises(ValueError):
        PostgresEventStorage()
