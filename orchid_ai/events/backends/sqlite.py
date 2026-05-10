"""SQLite implementations of the four event stores.

Each store ABC (signals / jobs / schedules / triggers) has its own
narrow concrete class — there is no super-class smushing four ABCs
into one Python type.  All four share a single
:class:`aiosqlite.Connection`, supplied by the :class:`SQLiteEventStorage`
facade that owns the lifecycle (open, run migrations, close).

The facade pattern matters because:

- Interfaces stay segregated (`ISP`): a producer that only writes
  signals depends on :class:`OrchidSignalStore`; a runner that only
  reads/writes job runs depends on :class:`OrchidJobStore`.
- The dispatcher's transactional outbox just needs the store and the
  queue to share a connection — the facade hands out the same
  ``aiosqlite.Connection`` to both, and the queue's
  ``transaction()`` returns a :class:`_SQLiteDBTransaction` that
  carries that connection.

JSON columns (``payload``, ``identity_claim``, ``result``,
``metadata``, ``config``, ``spec``, ``allowed_types``) are stored as
serialised TEXT.  The store owns the ``json.dumps`` / ``json.loads``
boundary — callers see plain Python ``dict`` / ``list``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import uuid as _uuid
from collections.abc import Iterable
from typing import Any, Sequence

import aiosqlite

from ...core.events.errors import SignalDuplicateError
from ...core.events.job import JobRun, JobSpec, JobStatus
from ...core.events.queue import DBTransaction
from ...core.events.signal import Signal
from ...core.events.store import (
    OrchidJobStore,
    OrchidScheduleRecord,
    OrchidScheduleStore,
    OrchidSignalStore,
    OrchidTriggerRecord,
    OrchidTriggerStore,
)
from ...persistence.sqlite import OrchidSQLiteMigrationRunner
from ..queues.sqlite import _SQLiteDBTransaction

_logger = logging.getLogger(__name__)


# ── Storage facade ──────────────────────────────────────────


class SQLiteEventStorage:
    """Owns the SQLite connection + migrations and exposes the four
    stores as attributes (``signals``, ``jobs``, ``schedules``,
    ``triggers``).

    Two construction modes:

    - ``conn=<aiosqlite.Connection>`` — share an externally-owned
      connection (typically the chat-storage connection).  The
      facade does NOT close the connection on :meth:`close`.
    - ``dsn=<path>`` — open a private connection.  Closed by the
      facade on :meth:`close`.

    ``init_db()`` is idempotent — it runs the framework migration
    runner regardless, which is a no-op when the schema is already
    up to date.
    """

    def __init__(
        self,
        *,
        conn: aiosqlite.Connection | None = None,
        dsn: str | None = None,
        extra_migrations_package: str | None = None,
    ) -> None:
        if (conn is None) == (dsn is None):
            raise ValueError("SQLiteEventStorage requires exactly one of conn= or dsn=")
        self._owned_conn = conn is None
        self._conn: aiosqlite.Connection | None = conn
        self._dsn = os.path.expanduser(dsn) if dsn is not None else None
        self._migrator = OrchidSQLiteMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

        # Stores are constructed lazily so ``init_db`` can wire them
        # to the now-open connection.  Tests can also inject a
        # pre-opened connection and skip ``init_db``.
        self._signals: SQLiteSignalStore | None = None
        self._jobs: SQLiteJobStore | None = None
        self._schedules: SQLiteScheduleStore | None = None
        self._triggers: SQLiteTriggerStore | None = None

    # ── Lifecycle ────────────────────────────────────────

    async def init_db(self) -> None:
        if self._conn is None:
            assert self._dsn is not None
            if self._dsn != ":memory:":
                os.makedirs(os.path.dirname(self._dsn) or ".", exist_ok=True)
            self._conn = await aiosqlite.connect(self._dsn)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
        # Idempotent — runs framework migrations including events.
        await self._migrator.run_up(self._conn)
        # Wire the stores now that we have a live connection.
        self._signals = SQLiteSignalStore(conn=self._conn)
        self._jobs = SQLiteJobStore(conn=self._conn)
        self._schedules = SQLiteScheduleStore(conn=self._conn)
        self._triggers = SQLiteTriggerStore(conn=self._conn)
        _logger.info(
            "[SQLiteEventStorage] Initialised — %s",
            self._dsn or "shared connection",
        )

    async def close(self) -> None:
        if self._owned_conn and self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ── Store accessors ──────────────────────────────────

    @property
    def signals(self) -> "SQLiteSignalStore":
        if self._signals is None:
            raise RuntimeError("SQLiteEventStorage used before init_db()")
        return self._signals

    @property
    def jobs(self) -> "SQLiteJobStore":
        if self._jobs is None:
            raise RuntimeError("SQLiteEventStorage used before init_db()")
        return self._jobs

    @property
    def schedules(self) -> "SQLiteScheduleStore":
        if self._schedules is None:
            raise RuntimeError("SQLiteEventStorage used before init_db()")
        return self._schedules

    @property
    def triggers(self) -> "SQLiteTriggerStore":
        if self._triggers is None:
            raise RuntimeError("SQLiteEventStorage used before init_db()")
        return self._triggers


# ── Signal store ─────────────────────────────────────────────


class SQLiteSignalStore(OrchidSignalStore):
    """Append-only persistence of :class:`Signal` rows."""

    def __init__(self, *, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def insert(self, signal: Signal, *, tx: DBTransaction | None = None) -> Signal:
        conn = _conn_from_tx(tx) or self._conn
        try:
            await conn.execute(
                "INSERT INTO signals "
                "(signal_id, type, source, payload, tenant_key, user_id, "
                " correlation_id, dedupe_key, identity_claim, chat_binding, "
                " occurred_at, persisted_at, relay_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(signal.signal_id),
                    signal.type,
                    signal.source,
                    json.dumps(signal.payload),
                    signal.tenant_key,
                    signal.user_id,
                    signal.correlation_id,
                    signal.dedupe_key,
                    json.dumps(signal.identity_claim) if signal.identity_claim else None,
                    json.dumps(signal.chat_binding) if signal.chat_binding else None,
                    signal.occurred_at.isoformat(),
                    signal.persisted_at.isoformat(),
                    signal.relay_status,
                ),
            )
        except aiosqlite.IntegrityError as exc:
            # ``signals_source_dedupe_idx`` is a partial unique index
            # over (source, dedupe_key) where dedupe_key is non-null.
            # Map that collision to ``SignalDuplicateError`` so the
            # dispatcher can surface ``deduplicated=True``.
            if signal.dedupe_key is not None:
                existing = await self.find_by_dedupe(source=signal.source, dedupe_key=signal.dedupe_key)
                raise SignalDuplicateError(str(existing) if existing else "") from exc
            raise
        if tx is None:
            await conn.commit()
        return signal

    async def get(self, signal_id: _uuid.UUID) -> Signal | None:
        cursor = await self._conn.execute("SELECT * FROM signals WHERE signal_id = ?", (str(signal_id),))
        row = await cursor.fetchone()
        return _row_to_signal(row) if row is not None else None

    async def find_by_dedupe(self, *, source: str, dedupe_key: str | None) -> _uuid.UUID | None:
        if dedupe_key is None:
            return None
        cursor = await self._conn.execute(
            "SELECT signal_id FROM signals  WHERE source = ? AND dedupe_key = ? LIMIT 1",
            (source, dedupe_key),
        )
        row = await cursor.fetchone()
        return _uuid.UUID(row["signal_id"]) if row is not None else None

    async def list(
        self,
        *,
        type: str | None = None,
        tenant_key: str | None = None,
        since: _dt.datetime | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        sql = "SELECT * FROM signals"
        clauses: list[str] = []
        params: list[Any] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if tenant_key is not None:
            clauses.append("tenant_key = ?")
            params.append(tenant_key)
        if since is not None:
            clauses.append("persisted_at >= ?")
            params.append(since.isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY persisted_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_signal(r) for r in rows]

    async def update_relay_status(self, signal_id: _uuid.UUID, *, status: str) -> None:
        await self._conn.execute(
            "UPDATE signals SET relay_status = ? WHERE signal_id = ?",
            (status, str(signal_id)),
        )
        await self._conn.commit()

    async def list_by_relay_status(self, *, status: str, limit: int = 100) -> list[Signal]:
        cursor = await self._conn.execute(
            "SELECT * FROM signals WHERE relay_status = ?  ORDER BY persisted_at ASC LIMIT ?",
            (status, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_signal(r) for r in rows]


# ── Job store ────────────────────────────────────────────────


class SQLiteJobStore(OrchidJobStore):
    """One row per attempted job run."""

    def __init__(self, *, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def insert(self, run: JobRun) -> JobRun:
        try:
            await self._conn.execute(
                "INSERT INTO job_runs "
                "(run_id, trigger_id, signal_id, attempt_number, status, "
                " agent_name, parallelism_key, spec, visibility, "
                " visibility_user_id, queued_at, started_at, finished_at, "
                " result, error, next_retry_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(run.run_id),
                    run.spec.trigger_id,
                    str(run.spec.signal_id),
                    run.attempt_number,
                    run.status.value,
                    run.spec.agent_name,
                    run.spec.parallelism_key,
                    json.dumps(_jobspec_to_dict(run.spec)),
                    run.spec.visibility,
                    run.spec.visibility_user_id,
                    run.queued_at.isoformat(),
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                    json.dumps(run.result) if run.result is not None else None,
                    run.error,
                    run.next_retry_at.isoformat() if run.next_retry_at else None,
                    json.dumps(run.metadata or {}),
                ),
            )
            await self._conn.commit()
            return run
        except aiosqlite.IntegrityError:
            # ``UNIQUE (trigger_id, signal_id, attempt_number)`` —
            # a redelivered queue message tried to insert this run
            # again.  Return the persisted one (idempotent contract).
            existing = await self._fetch_run_by_dedupe(
                trigger_id=run.spec.trigger_id,
                signal_id=run.spec.signal_id,
                attempt_number=run.attempt_number,
            )
            if existing is not None:
                return existing
            raise

    async def update(self, run: JobRun) -> None:
        await self._conn.execute(
            "UPDATE job_runs "
            "   SET status = ?, started_at = ?, finished_at = ?, "
            "       result = ?, error = ?, next_retry_at = ?, metadata = ? "
            " WHERE run_id = ?",
            (
                run.status.value,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                json.dumps(run.result) if run.result is not None else None,
                run.error,
                run.next_retry_at.isoformat() if run.next_retry_at else None,
                json.dumps(run.metadata or {}),
                str(run.run_id),
            ),
        )
        await self._conn.commit()

    async def get(self, run_id: _uuid.UUID) -> JobRun | None:
        cursor = await self._conn.execute("SELECT * FROM job_runs WHERE run_id = ?", (str(run_id),))
        row = await cursor.fetchone()
        return _row_to_run(row) if row is not None else None

    async def list(
        self,
        *,
        trigger_id: str | None = None,
        status: str | None = None,
        statuses: Sequence[str] | None = None,
        since: _dt.datetime | None = None,
        limit: int = 100,
        chat_binding_chat_id: str | None = None,
    ) -> list[JobRun]:
        sql = "SELECT * FROM job_runs"
        clauses: list[str] = []
        params: list[Any] = []
        if trigger_id is not None:
            clauses.append("trigger_id = ?")
            params.append(trigger_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if statuses is not None:
            statuses_list = list(statuses)
            placeholders = ",".join("?" * len(statuses_list)) or "NULL"
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses_list)
        if since is not None:
            clauses.append("queued_at >= ?")
            params.append(since.isoformat())
        if chat_binding_chat_id is not None:
            # SQLite's JSON1 ``json_extract`` reads the chat_id field
            # out of the spec column without a sidecar column —
            # mirrors the Postgres JSONB path.
            clauses.append("json_extract(spec, '$.chat_binding.chat_id') = ?")
            params.append(chat_binding_chat_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY queued_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_run(r) for r in rows]

    async def latest_attempt(self, *, trigger_id: str, signal_id: _uuid.UUID) -> int:
        cursor = await self._conn.execute(
            "SELECT MAX(attempt_number) AS n   FROM job_runs  WHERE trigger_id = ? AND signal_id = ?",
            (trigger_id, str(signal_id)),
        )
        row = await cursor.fetchone()
        if row is None or row["n"] is None:
            return 0
        return int(row["n"])

    async def find_latest(self, *, trigger_id: str, signal_id: _uuid.UUID) -> JobRun | None:
        cursor = await self._conn.execute(
            "SELECT * FROM job_runs  WHERE trigger_id = ? AND signal_id = ?  ORDER BY attempt_number DESC LIMIT 1",
            (trigger_id, str(signal_id)),
        )
        row = await cursor.fetchone()
        return _row_to_run(row) if row is not None else None

    async def _fetch_run_by_dedupe(
        self, *, trigger_id: str, signal_id: _uuid.UUID, attempt_number: int
    ) -> JobRun | None:
        cursor = await self._conn.execute(
            "SELECT * FROM job_runs  WHERE trigger_id = ? AND signal_id = ? AND attempt_number = ?",
            (trigger_id, str(signal_id), attempt_number),
        )
        row = await cursor.fetchone()
        return _row_to_run(row) if row is not None else None


# ── Schedule store ───────────────────────────────────────────


class SQLiteScheduleStore(OrchidScheduleStore):
    """Cron / interval entries consumed by the scheduler producer."""

    def __init__(self, *, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def upsert(self, record: OrchidScheduleRecord) -> None:
        await self._conn.execute(
            "INSERT OR REPLACE INTO schedules "
            "(schedule_id, trigger_id, cron, interval_seconds, identity_claim, "
            " last_fire_at, next_fire_at, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.schedule_id,
                record.trigger_id,
                record.cron,
                record.interval_seconds,
                json.dumps(record.identity_claim),
                record.last_fire_at.isoformat() if record.last_fire_at else None,
                record.next_fire_at.isoformat() if record.next_fire_at else None,
                1 if record.enabled else 0,
            ),
        )
        await self._conn.commit()

    async def get(self, schedule_id: str) -> OrchidScheduleRecord | None:
        cursor = await self._conn.execute("SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,))
        row = await cursor.fetchone()
        return _row_to_schedule(row) if row is not None else None

    async def list(self) -> Iterable[OrchidScheduleRecord]:
        cursor = await self._conn.execute("SELECT * FROM schedules")
        rows = await cursor.fetchall()
        return [_row_to_schedule(r) for r in rows]

    async def set_enabled(self, schedule_id: str, *, enabled: bool) -> None:
        await self._conn.execute(
            "UPDATE schedules SET enabled = ? WHERE schedule_id = ?",
            (1 if enabled else 0, schedule_id),
        )
        await self._conn.commit()

    async def record_fire(
        self,
        schedule_id: str,
        *,
        last_fire_at: _dt.datetime,
        next_fire_at: _dt.datetime | None,
    ) -> None:
        await self._conn.execute(
            "UPDATE schedules    SET last_fire_at = ?, next_fire_at = ?  WHERE schedule_id = ?",
            (
                last_fire_at.isoformat(),
                next_fire_at.isoformat() if next_fire_at else None,
                schedule_id,
            ),
        )
        await self._conn.commit()


# ── Trigger store ────────────────────────────────────────────


class SQLiteTriggerStore(OrchidTriggerStore):
    """Versioned snapshots of trigger configs."""

    def __init__(self, *, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def insert_version(self, record: OrchidTriggerRecord) -> None:
        await self._conn.execute(
            "INSERT INTO triggers (trigger_id, version, config, deleted_at, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                record.trigger_id,
                record.version,
                json.dumps(record.config),
                record.deleted_at.isoformat() if record.deleted_at else None,
                record.created_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def latest(self, trigger_id: str) -> OrchidTriggerRecord | None:
        cursor = await self._conn.execute(
            "SELECT * FROM triggers  WHERE trigger_id = ? AND deleted_at IS NULL  ORDER BY version DESC LIMIT 1",
            (trigger_id,),
        )
        row = await cursor.fetchone()
        return _row_to_trigger(row) if row is not None else None

    async def list_active(self) -> Iterable[OrchidTriggerRecord]:
        cursor = await self._conn.execute(
            "SELECT t.* FROM triggers t "
            "  WHERE t.deleted_at IS NULL "
            "    AND t.version = (SELECT MAX(version) FROM triggers t2 "
            "                       WHERE t2.trigger_id = t.trigger_id "
            "                         AND t2.deleted_at IS NULL)"
        )
        rows = await cursor.fetchall()
        return [_row_to_trigger(r) for r in rows]

    async def soft_delete(self, trigger_id: str, *, deleted_at: _dt.datetime) -> None:
        await self._conn.execute(
            "UPDATE triggers SET deleted_at = ?  WHERE trigger_id = ? AND deleted_at IS NULL",
            (deleted_at.isoformat(), trigger_id),
        )
        await self._conn.commit()


# ── Helpers ──────────────────────────────────────────────────


def _conn_from_tx(tx: DBTransaction | None) -> aiosqlite.Connection | None:
    if tx is None:
        return None
    if isinstance(tx, _SQLiteDBTransaction):
        return tx.conn
    return None


def _row_to_signal(row: aiosqlite.Row) -> Signal:
    return Signal(
        type=row["type"],
        payload=json.loads(row["payload"]),
        source=row["source"],
        occurred_at=_dt.datetime.fromisoformat(row["occurred_at"]),
        tenant_key=row["tenant_key"],
        signal_id=_uuid.UUID(row["signal_id"]),
        persisted_at=_dt.datetime.fromisoformat(row["persisted_at"]),
        user_id=row["user_id"],
        correlation_id=row["correlation_id"],
        dedupe_key=row["dedupe_key"],
        identity_claim=json.loads(row["identity_claim"]) if row["identity_claim"] else None,
        chat_binding=json.loads(row["chat_binding"]) if row["chat_binding"] else None,
        relay_status=row["relay_status"],
    )


def _row_to_run(row: aiosqlite.Row) -> JobRun:
    spec_dict = json.loads(row["spec"])
    spec = JobSpec(
        trigger_id=spec_dict["trigger_id"],
        signal_id=_uuid.UUID(spec_dict["signal_id"]),
        agent_name=spec_dict["agent_name"],
        prompt=spec_dict["prompt"],
        identity_claim=spec_dict["identity_claim"],
        correlation_id=spec_dict.get("correlation_id"),
        parallelism_key=spec_dict["parallelism_key"],
        visibility=spec_dict.get("visibility", row["visibility"]),
        visibility_user_id=spec_dict.get("visibility_user_id", row["visibility_user_id"]),
        chat_binding=spec_dict.get("chat_binding"),
    )
    return JobRun(
        run_id=_uuid.UUID(row["run_id"]),
        spec=spec,
        attempt_number=int(row["attempt_number"]),
        status=JobStatus(row["status"]),
        queued_at=_dt.datetime.fromisoformat(row["queued_at"]),
        started_at=_dt.datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=_dt.datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        next_retry_at=_dt.datetime.fromisoformat(row["next_retry_at"]) if row["next_retry_at"] else None,
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )


def _row_to_schedule(row: aiosqlite.Row) -> OrchidScheduleRecord:
    return OrchidScheduleRecord(
        schedule_id=row["schedule_id"],
        trigger_id=row["trigger_id"],
        cron=row["cron"],
        interval_seconds=row["interval_seconds"],
        identity_claim=json.loads(row["identity_claim"]),
        last_fire_at=_dt.datetime.fromisoformat(row["last_fire_at"]) if row["last_fire_at"] else None,
        next_fire_at=_dt.datetime.fromisoformat(row["next_fire_at"]) if row["next_fire_at"] else None,
        enabled=bool(row["enabled"]),
    )


def _row_to_trigger(row: aiosqlite.Row) -> OrchidTriggerRecord:
    return OrchidTriggerRecord(
        trigger_id=row["trigger_id"],
        version=int(row["version"]),
        config=json.loads(row["config"]),
        created_at=_dt.datetime.fromisoformat(row["created_at"]),
        deleted_at=_dt.datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
    )


def _jobspec_to_dict(spec: JobSpec) -> dict[str, Any]:
    return {
        "trigger_id": spec.trigger_id,
        "signal_id": str(spec.signal_id),
        "agent_name": spec.agent_name,
        "prompt": spec.prompt,
        "identity_claim": spec.identity_claim,
        "correlation_id": spec.correlation_id,
        "parallelism_key": spec.parallelism_key,
        "visibility": spec.visibility,
        "visibility_user_id": spec.visibility_user_id,
        "chat_binding": spec.chat_binding,
    }
