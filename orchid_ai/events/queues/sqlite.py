"""SQLite-backed signal queue.

Implements :class:`OrchidSignalQueue` against the ``signal_queue`` and
``signal_queue_dead_letter`` tables created by the unified ``v001`` migration.

Two construction modes:

- ``conn=<aiosqlite.Connection>`` — share the connection with
  :class:`SQLiteEventStore` (and the chat storage) so the dispatcher's
  transactional outbox can commit the ``signals`` insert and the
  ``signal_queue`` insert atomically.  This is the production wiring.
- ``dsn=<path>`` — open a private connection.  Convenient for tests
  and single-purpose tools.

Either way ``init_db()`` runs the framework's migration runner against
the connection so the events tables exist before the first ``enqueue``.
The runner is idempotent — it's safe to call ``init_db`` on a DB that
already has v001 applied (e.g. because the chat storage went
first).

SQLite is a single-writer engine, so the queue uses a per-instance
:class:`asyncio.Lock` to serialise the multi-statement dequeue/ack/nack
flows.  ``transaction()`` opens a deferred transaction (``BEGIN``) and
commits on exit; on exception it rolls back and re-raises so the
dispatcher's outbox semantics hold (signal insert + enqueue are
atomic).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

from ...core.events.queue import (
    DBTransaction,
    OrchidSignalQueue,
    QueuedSignal,
)
from ...persistence.sqlite import OrchidSQLiteMigrationRunner

_logger = logging.getLogger(__name__)


class _SQLiteDBTransaction(DBTransaction):
    """Concrete ``DBTransaction`` carrying the connection back to the
    store / queue so they can run their writes inside the same
    transaction.  Opaque to the dispatcher, narrow to the backend."""

    __slots__ = ("conn",)

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn


class SQLiteSignalQueue(OrchidSignalQueue):
    """Single-process SQLite queue with leases + dead-letter."""

    def __init__(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
        dsn: str | None = None,
        extra_migrations_package: str | None = None,
        max_attempts: int = 5,
    ) -> None:
        if (conn is None) == (dsn is None):
            raise ValueError("SQLiteSignalQueue requires exactly one of conn= or dsn=")
        self._owned_conn = conn is None
        self._conn: aiosqlite.Connection | None = conn
        self._dsn = os.path.expanduser(dsn) if dsn is not None else None
        self._max_attempts = max_attempts
        self._migrator = OrchidSQLiteMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )
        self._tx_lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        if self._conn is None:
            assert self._dsn is not None
            if self._dsn != ":memory:":
                os.makedirs(os.path.dirname(self._dsn) or ".", exist_ok=True)
            self._conn = await aiosqlite.connect(self._dsn)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
        # Migration runner is idempotent — safe to call when the chat
        # storage already ran it.
        await self._migrator.run_up(self._conn)

    async def close(self) -> None:
        if self._owned_conn and self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteSignalQueue used before init_db() — connection is None")
        return self._conn

    # ── Transaction (outbox boundary) ────────────────────────

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DBTransaction | None]:
        conn = self.connection
        # Serialise writers — SQLite only allows one at a time, and
        # nesting BEGIN within an open transaction errors.
        await self._tx_lock.acquire()
        try:
            await conn.execute("BEGIN")
            tx = _SQLiteDBTransaction(conn)
            try:
                yield tx
            except BaseException:
                await conn.execute("ROLLBACK")
                raise
            else:
                await conn.commit()
        finally:
            self._tx_lock.release()

    # ── Queue operations ─────────────────────────────────────

    async def enqueue(
        self,
        signal_id: _uuid.UUID,
        *,
        priority: int = 0,
        tx: DBTransaction | None = None,
    ) -> str:
        msg_id = str(_uuid.uuid4())
        now_iso = _now_iso()
        conn = _conn_from_tx(tx) or self.connection

        # When the dispatcher already opened a transaction, do the
        # write on its conn without committing — the outer scope will
        # commit.  Otherwise, run our own self-contained statement
        # (SQLite auto-commits non-transactional writes).
        await conn.execute(
            "INSERT INTO signal_queue "
            "(queue_msg_id, signal_id, priority, enqueued_at, visible_after, "
            " lease_until, attempt) "
            "VALUES (?, ?, ?, ?, ?, NULL, 0)",
            (msg_id, str(signal_id), priority, now_iso, now_iso),
        )
        if tx is None:
            await conn.commit()
        return msg_id

    async def dequeue(self, *, batch_size: int, lease_seconds: int) -> list[QueuedSignal]:
        conn = self.connection
        async with self._tx_lock:
            now = _now_dt()
            now_iso = now.isoformat()
            new_lease = (now + _dt.timedelta(seconds=lease_seconds)).isoformat()

            # Two-phase under one transaction:
            #   1) SELECT visible candidate rows with priority desc + FIFO.
            #   2) UPDATE them with the new lease + attempt+=1.
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await conn.execute(
                    "SELECT queue_msg_id, signal_id, enqueued_at, attempt "
                    "  FROM signal_queue "
                    " WHERE visible_after <= ? "
                    "   AND (lease_until IS NULL OR lease_until <= ?) "
                    " ORDER BY priority DESC, enqueued_at "
                    " LIMIT ?",
                    (now_iso, now_iso, batch_size),
                )
                rows = await cursor.fetchall()
                if not rows:
                    await conn.commit()
                    return []

                msg_ids = [r["queue_msg_id"] for r in rows]
                placeholders = ",".join(["?"] * len(msg_ids))
                await conn.execute(
                    f"UPDATE signal_queue "
                    f"   SET lease_until = ?, attempt = attempt + 1 "
                    f" WHERE queue_msg_id IN ({placeholders})",
                    [new_lease, *msg_ids],
                )
                await conn.commit()
            except BaseException:
                await conn.execute("ROLLBACK")
                raise

            return [
                QueuedSignal(
                    queue_msg_id=r["queue_msg_id"],
                    signal_id=_uuid.UUID(r["signal_id"]),
                    enqueued_at=_dt.datetime.fromisoformat(r["enqueued_at"]),
                    lease_until=now + _dt.timedelta(seconds=lease_seconds),
                    attempt=int(r["attempt"]) + 1,
                )
                for r in rows
            ]

    async def ack(self, queue_msg_id: str) -> None:
        conn = self.connection
        await conn.execute(
            "DELETE FROM signal_queue WHERE queue_msg_id = ?",
            (queue_msg_id,),
        )
        await conn.commit()

    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        conn = self.connection
        async with self._tx_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await conn.execute(
                    "SELECT signal_id, attempt   FROM signal_queue  WHERE queue_msg_id = ?",
                    (queue_msg_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    await conn.commit()
                    return
                attempt = int(row["attempt"])
                signal_id = row["signal_id"]

                if attempt >= self._max_attempts:
                    await self._move_to_dlq_locked(
                        conn,
                        queue_msg_id=queue_msg_id,
                        signal_id=signal_id,
                        attempts=attempt,
                        reason="max attempts exceeded",
                    )
                    await conn.commit()
                    return

                visible_after = (_now_dt() + _dt.timedelta(seconds=retry_after_seconds)).isoformat()
                await conn.execute(
                    "UPDATE signal_queue    SET lease_until = NULL, visible_after = ?  WHERE queue_msg_id = ?",
                    (visible_after, queue_msg_id),
                )
                await conn.commit()
            except BaseException:
                await conn.execute("ROLLBACK")
                raise

    async def dead_letter(self, queue_msg_id: str, *, reason: str) -> None:
        conn = self.connection
        async with self._tx_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await conn.execute(
                    "SELECT signal_id, attempt   FROM signal_queue  WHERE queue_msg_id = ?",
                    (queue_msg_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    await conn.commit()
                    return
                await self._move_to_dlq_locked(
                    conn,
                    queue_msg_id=queue_msg_id,
                    signal_id=row["signal_id"],
                    attempts=int(row["attempt"]),
                    reason=reason,
                )
                await conn.commit()
            except BaseException:
                await conn.execute("ROLLBACK")
                raise

    # ── Helpers ──────────────────────────────────────────────

    async def _move_to_dlq_locked(
        self,
        conn: aiosqlite.Connection,
        *,
        queue_msg_id: str,
        signal_id: str,
        attempts: int,
        reason: str,
    ) -> None:
        await conn.execute(
            "INSERT OR REPLACE INTO signal_queue_dead_letter "
            "(queue_msg_id, signal_id, reason, failed_at, attempts) "
            "VALUES (?, ?, ?, ?, ?)",
            (queue_msg_id, signal_id, reason, _now_iso(), attempts),
        )
        await conn.execute(
            "DELETE FROM signal_queue WHERE queue_msg_id = ?",
            (queue_msg_id,),
        )

    # ── Test / observability helpers ─────────────────────────

    async def visible_count(self) -> int:
        cursor = await self.connection.execute(
            "SELECT COUNT(*) AS n "
            "  FROM signal_queue "
            " WHERE visible_after <= ? "
            "   AND (lease_until IS NULL OR lease_until <= ?)",
            (_now_iso(), _now_iso()),
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row is not None else 0

    async def in_flight_count(self) -> int:
        cursor = await self.connection.execute(
            "SELECT COUNT(*) AS n   FROM signal_queue  WHERE lease_until IS NOT NULL AND lease_until > ?",
            (_now_iso(),),
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row is not None else 0

    async def dead_letter_count(self) -> int:
        cursor = await self.connection.execute("SELECT COUNT(*) AS n FROM signal_queue_dead_letter")
        row = await cursor.fetchone()
        return int(row["n"]) if row is not None else 0


# ── Module-level helpers ────────────────────────────────────


def _conn_from_tx(tx: DBTransaction | None) -> aiosqlite.Connection | None:
    if tx is None:
        return None
    if isinstance(tx, _SQLiteDBTransaction):
        return tx.conn
    return None


def _now_dt() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _now_iso() -> str:
    return _now_dt().isoformat()
