"""Postgres-backed signal queue.

Implements :class:`OrchidSignalQueue` against the ``signal_queue`` and
``signal_queue_dead_letter`` tables created by the unified migration.
The dequeue path uses ``FOR UPDATE SKIP LOCKED`` so multiple worker
processes can drain the queue without coordination overhead.

Two construction modes mirror the SQLite backend:

- ``pool=<asyncpg.Pool>`` — share the chat-storage pool so signal +
  queue inserts commit in the same transaction (the dispatcher's
  outbox).  Production wiring.
- ``dsn=<string>`` — open a private pool.  Convenient for tests and
  ops tools that don't share a runtime with the chat storage.

After the dispatcher's outbox transaction commits, ``enqueue`` fires
``pg_notify('signal_queue', queue_msg_id::text)`` so workers can
react with sub-poll-interval latency (opt-out via
``notify_enabled=False``).  The notify happens on the same connection
that did the insert, *after* the outer transaction has committed,
because Postgres delivers ``NOTIFY`` payloads at commit time anyway —
queueing them inside the transaction is correct AND lets workers act
on a guaranteed-visible row.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from ...core.events.queue import (
    DBTransaction,
    OrchidSignalQueue,
    QueuedSignal,
)
from ...persistence.postgres import OrchidPostgresMigrationRunner

_logger = logging.getLogger(__name__)


class _PostgresDBTransaction(DBTransaction):
    """Concrete ``DBTransaction`` carrying the asyncpg connection back
    to the store / queue so they run their writes inside the same
    transaction.  Also collects the ``queue_msg_id``s that need a
    post-commit ``NOTIFY``."""

    __slots__ = ("conn", "pending_notifies")

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn
        self.pending_notifies: list[str] = []


class PostgresSignalQueue(OrchidSignalQueue):
    """asyncpg-backed durable queue with leases + dead-letter."""

    def __init__(
        self,
        *,
        pool: asyncpg.Pool | None = None,
        dsn: str | None = None,
        extra_migrations_package: str | None = None,
        max_attempts: int = 5,
        notify_enabled: bool = True,
        notify_channel: str = "signal_queue",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> None:
        if (pool is None) == (dsn is None):
            raise ValueError("PostgresSignalQueue requires exactly one of pool= or dsn=")
        self._owned_pool = pool is None
        self._pool: asyncpg.Pool | None = pool
        self._dsn = dsn
        self._max_attempts = max_attempts
        self._notify_enabled = notify_enabled
        self._notify_channel = notify_channel
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._migrator = OrchidPostgresMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        if self._pool is None:
            assert self._dsn is not None
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)

    async def close(self) -> None:
        if self._owned_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresSignalQueue used before init_db() — pool is None")
        return self._pool

    # ── Transaction (outbox boundary) ────────────────────────

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DBTransaction | None]:
        pool = self.pool
        async with pool.acquire() as conn:
            tx = _PostgresDBTransaction(conn)
            async with conn.transaction():
                yield tx
            # Outer transaction has now committed.  Fire any
            # post-commit NOTIFYs collected by enqueue().
            if self._notify_enabled and tx.pending_notifies:
                for msg_id in tx.pending_notifies:
                    try:
                        await conn.execute(
                            "SELECT pg_notify($1, $2)",
                            self._notify_channel,
                            msg_id,
                        )
                    except Exception:  # pragma: no cover — notify is best-effort
                        _logger.exception(
                            "pg_notify failed on channel %s",
                            self._notify_channel,
                        )

    # ── Queue operations ─────────────────────────────────────

    async def enqueue(
        self,
        signal_id: _uuid.UUID,
        *,
        priority: int = 0,
        tx: DBTransaction | None = None,
    ) -> str:
        msg_id = _uuid.uuid4()
        if tx is None:
            # Stand-alone enqueue — open a short-lived transaction so
            # the (insert + notify) pair stays atomic.
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO signal_queue "
                        "(queue_msg_id, signal_id, priority, enqueued_at, "
                        " visible_after, lease_until, attempt) "
                        "VALUES ($1, $2, $3, now(), now(), NULL, 0)",
                        msg_id,
                        signal_id,
                        priority,
                    )
                if self._notify_enabled:
                    try:
                        await conn.execute(
                            "SELECT pg_notify($1, $2)",
                            self._notify_channel,
                            str(msg_id),
                        )
                    except Exception:  # pragma: no cover
                        _logger.exception(
                            "pg_notify failed on channel %s",
                            self._notify_channel,
                        )
            return str(msg_id)

        # Inside the dispatcher's outbox transaction — write through
        # the supplied tx and stage the NOTIFY for after commit.
        conn = _conn_from_tx(tx)
        if conn is None:
            raise RuntimeError(
                "PostgresSignalQueue.enqueue received a non-Postgres DBTransaction; outbox boundary mismatch"
            )
        await conn.execute(
            "INSERT INTO signal_queue "
            "(queue_msg_id, signal_id, priority, enqueued_at, "
            " visible_after, lease_until, attempt) "
            "VALUES ($1, $2, $3, now(), now(), NULL, 0)",
            msg_id,
            signal_id,
            priority,
        )
        if isinstance(tx, _PostgresDBTransaction):
            tx.pending_notifies.append(str(msg_id))
        return str(msg_id)

    async def dequeue(self, *, batch_size: int, lease_seconds: int) -> list[QueuedSignal]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    UPDATE signal_queue
                       SET lease_until = now() + ($1 || ' seconds')::interval,
                           attempt     = attempt + 1
                     WHERE queue_msg_id IN (
                         SELECT queue_msg_id FROM signal_queue
                          WHERE visible_after <= now()
                            AND (lease_until IS NULL OR lease_until <= now())
                          ORDER BY priority DESC, enqueued_at
                          FOR UPDATE SKIP LOCKED
                          LIMIT $2
                     )
                    RETURNING queue_msg_id, signal_id, enqueued_at,
                              lease_until, attempt
                    """,
                    str(lease_seconds),
                    batch_size,
                )
        return [
            QueuedSignal(
                queue_msg_id=str(r["queue_msg_id"]),
                signal_id=r["signal_id"],
                enqueued_at=_ensure_aware(r["enqueued_at"]),
                lease_until=_ensure_aware(r["lease_until"]),
                attempt=int(r["attempt"]),
            )
            for r in rows
        ]

    async def ack(self, queue_msg_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM signal_queue WHERE queue_msg_id = $1",
                _uuid.UUID(queue_msg_id),
            )

    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT signal_id, attempt FROM signal_queue  WHERE queue_msg_id = $1",
                    _uuid.UUID(queue_msg_id),
                )
                if row is None:
                    return
                attempt = int(row["attempt"])
                if attempt >= self._max_attempts:
                    await self._move_to_dlq_locked(
                        conn,
                        queue_msg_id=queue_msg_id,
                        signal_id=row["signal_id"],
                        attempts=attempt,
                        reason="max attempts exceeded",
                    )
                    return
                await conn.execute(
                    "UPDATE signal_queue "
                    "   SET lease_until = NULL, "
                    "       visible_after = now() + ($1 || ' seconds')::interval "
                    " WHERE queue_msg_id = $2",
                    str(retry_after_seconds),
                    _uuid.UUID(queue_msg_id),
                )

    async def dead_letter(self, queue_msg_id: str, *, reason: str) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT signal_id, attempt FROM signal_queue  WHERE queue_msg_id = $1",
                    _uuid.UUID(queue_msg_id),
                )
                if row is None:
                    return
                await self._move_to_dlq_locked(
                    conn,
                    queue_msg_id=queue_msg_id,
                    signal_id=row["signal_id"],
                    attempts=int(row["attempt"]),
                    reason=reason,
                )

    # ── Helpers ──────────────────────────────────────────────

    async def _move_to_dlq_locked(
        self,
        conn: asyncpg.Connection,
        *,
        queue_msg_id: str,
        signal_id: _uuid.UUID,
        attempts: int,
        reason: str,
    ) -> None:
        await conn.execute(
            "INSERT INTO signal_queue_dead_letter "
            "(queue_msg_id, signal_id, reason, failed_at, attempts) "
            "VALUES ($1, $2, $3, now(), $4) "
            "ON CONFLICT (queue_msg_id) DO NOTHING",
            _uuid.UUID(queue_msg_id),
            signal_id,
            reason,
            attempts,
        )
        await conn.execute(
            "DELETE FROM signal_queue WHERE queue_msg_id = $1",
            _uuid.UUID(queue_msg_id),
        )

    # ── Test / observability helpers ─────────────────────────

    async def visible_count(self) -> int:
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM signal_queue "
                    " WHERE visible_after <= now() "
                    "   AND (lease_until IS NULL OR lease_until <= now())"
                )
            )

    async def in_flight_count(self) -> int:
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM signal_queue  WHERE lease_until IS NOT NULL AND lease_until > now()"
                )
            )

    async def dead_letter_count(self) -> int:
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval("SELECT COUNT(*) FROM signal_queue_dead_letter"))


def _conn_from_tx(tx: DBTransaction) -> asyncpg.Connection | None:
    if isinstance(tx, _PostgresDBTransaction):
        return tx.conn
    return None


def _ensure_aware(value: _dt.datetime | None) -> _dt.datetime:
    """asyncpg returns ``TIMESTAMPTZ`` as tz-aware; some test setups
    that mock this might hand us a naive datetime — tag UTC defensively
    so downstream comparisons work."""
    if value is None:
        return _dt.datetime.now(tz=_dt.UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.UTC)
    return value
