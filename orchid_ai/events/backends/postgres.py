"""Postgres implementations of the four event stores.

Mirrors the SQLite layout: a :class:`PostgresEventStorage` facade
opens / closes the pool and exposes four narrow store classes
(``signals``, ``jobs``, ``schedules``, ``triggers``) that share the
same :class:`asyncpg.Pool`.

The pool can be supplied externally (so events shares the chat-storage
pool) or owned by the facade (for tests / standalone tooling).

JSON columns travel as serialised strings — asyncpg's default codec
implicitly casts ``text → jsonb`` on the receiving side, the same
pattern :class:`OrchidPostgresChatStorage` uses for ``chat_messages
.agents_used`` etc.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid as _uuid
from collections.abc import Iterable
from typing import Any, Sequence

import asyncpg

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
from ...persistence.postgres import OrchidPostgresMigrationRunner
from ..queues.postgres import _PostgresDBTransaction

_logger = logging.getLogger(__name__)


# ── Storage facade ──────────────────────────────────────────


class PostgresEventStorage:
    """Owns the pool + migrations and exposes the four stores."""

    def __init__(
        self,
        *,
        pool: asyncpg.Pool | None = None,
        dsn: str | None = None,
        extra_migrations_package: str | None = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        if (pool is None) == (dsn is None):
            raise ValueError("PostgresEventStorage requires exactly one of pool= or dsn=")
        self._owned_pool = pool is None
        self._pool: asyncpg.Pool | None = pool
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._migrator = OrchidPostgresMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )
        self._signals: PostgresSignalStore | None = None
        self._jobs: PostgresJobStore | None = None
        self._schedules: PostgresScheduleStore | None = None
        self._triggers: PostgresTriggerStore | None = None

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
        self._signals = PostgresSignalStore(pool=self._pool)
        self._jobs = PostgresJobStore(pool=self._pool)
        self._schedules = PostgresScheduleStore(pool=self._pool)
        self._triggers = PostgresTriggerStore(pool=self._pool)
        safe_dsn = self._dsn.split("@")[-1] if self._dsn and "@" in self._dsn else "shared pool"
        _logger.info("[PostgresEventStorage] Initialised — %s", safe_dsn)

    async def close(self) -> None:
        if self._owned_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def signals(self) -> "PostgresSignalStore":
        if self._signals is None:
            raise RuntimeError("PostgresEventStorage used before init_db()")
        return self._signals

    @property
    def jobs(self) -> "PostgresJobStore":
        if self._jobs is None:
            raise RuntimeError("PostgresEventStorage used before init_db()")
        return self._jobs

    @property
    def schedules(self) -> "PostgresScheduleStore":
        if self._schedules is None:
            raise RuntimeError("PostgresEventStorage used before init_db()")
        return self._schedules

    @property
    def triggers(self) -> "PostgresTriggerStore":
        if self._triggers is None:
            raise RuntimeError("PostgresEventStorage used before init_db()")
        return self._triggers


# ── Signal store ─────────────────────────────────────────────


class PostgresSignalStore(OrchidSignalStore):
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(self, signal: Signal, *, tx: DBTransaction | None = None) -> Signal:
        sql = (
            "INSERT INTO signals "
            "(signal_id, type, source, payload, tenant_key, user_id, "
            " correlation_id, dedupe_key, identity_claim, chat_binding, "
            " occurred_at, persisted_at, relay_status) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9::jsonb, "
            "        $10::jsonb, $11, $12, $13)"
        )
        params = (
            signal.signal_id,
            signal.type,
            signal.source,
            json.dumps(signal.payload),
            signal.tenant_key,
            signal.user_id,
            signal.correlation_id,
            signal.dedupe_key,
            json.dumps(signal.identity_claim) if signal.identity_claim else None,
            json.dumps(signal.chat_binding) if signal.chat_binding else None,
            signal.occurred_at,
            signal.persisted_at,
            signal.relay_status,
        )
        try:
            if tx is not None:
                conn = _conn_from_tx(tx)
                if conn is None:
                    raise RuntimeError("PostgresSignalStore.insert: non-Postgres DBTransaction supplied")
                await conn.execute(sql, *params)
            else:
                async with self._pool.acquire() as conn:
                    await conn.execute(sql, *params)
        except asyncpg.UniqueViolationError as exc:
            if signal.dedupe_key is not None:
                existing = await self.find_by_dedupe(source=signal.source, dedupe_key=signal.dedupe_key)
                raise SignalDuplicateError(str(existing) if existing else "") from exc
            raise
        return signal

    async def get(self, signal_id: _uuid.UUID) -> Signal | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM signals WHERE signal_id = $1", signal_id)
        return _row_to_signal(row) if row is not None else None

    async def find_by_dedupe(self, *, source: str, dedupe_key: str | None) -> _uuid.UUID | None:
        if dedupe_key is None:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT signal_id FROM signals  WHERE source = $1 AND dedupe_key = $2 LIMIT 1",
                source,
                dedupe_key,
            )
        return row["signal_id"] if row is not None else None

    async def list(
        self,
        *,
        type: str | None = None,
        tenant_key: str | None = None,
        since: _dt.datetime | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        clauses: list[str] = []
        params: list[Any] = []
        if type is not None:
            clauses.append(f"type = ${len(params) + 1}")
            params.append(type)
        if tenant_key is not None:
            clauses.append(f"tenant_key = ${len(params) + 1}")
            params.append(tenant_key)
        if since is not None:
            clauses.append(f"persisted_at >= ${len(params) + 1}")
            params.append(since)
        sql = "SELECT * FROM signals"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY persisted_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_signal(r) for r in rows]

    async def update_relay_status(self, signal_id: _uuid.UUID, *, status: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE signals SET relay_status = $1 WHERE signal_id = $2",
                status,
                signal_id,
            )

    async def list_by_relay_status(self, *, status: str, limit: int = 100) -> list[Signal]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM signals WHERE relay_status = $1  ORDER BY persisted_at ASC LIMIT $2",
                status,
                limit,
            )
        return [_row_to_signal(r) for r in rows]


# ── Job store ────────────────────────────────────────────────


class PostgresJobStore(OrchidJobStore):
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(self, run: JobRun) -> JobRun:
        sql = (
            "INSERT INTO job_runs "
            "(run_id, trigger_id, signal_id, attempt_number, status, "
            " agent_name, parallelism_key, spec, visibility, "
            " visibility_user_id, queued_at, started_at, finished_at, "
            " result, error, next_retry_at, metadata) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, "
            "        $11, $12, $13, $14::jsonb, $15, $16, $17::jsonb)"
        )
        params = (
            run.run_id,
            run.spec.trigger_id,
            run.spec.signal_id,
            run.attempt_number,
            run.status.value,
            run.spec.agent_name,
            run.spec.parallelism_key,
            json.dumps(_jobspec_to_dict(run.spec)),
            run.spec.visibility,
            run.spec.visibility_user_id,
            run.queued_at,
            run.started_at,
            run.finished_at,
            json.dumps(run.result) if run.result is not None else None,
            run.error,
            run.next_retry_at,
            json.dumps(run.metadata or {}),
        )
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(sql, *params)
            return run
        except asyncpg.UniqueViolationError:
            existing = await self._fetch_run_by_dedupe(
                trigger_id=run.spec.trigger_id,
                signal_id=run.spec.signal_id,
                attempt_number=run.attempt_number,
            )
            if existing is not None:
                return existing
            raise

    async def update(self, run: JobRun) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE job_runs "
                "   SET status = $1, started_at = $2, finished_at = $3, "
                "       result = $4::jsonb, error = $5, next_retry_at = $6, "
                "       metadata = $7::jsonb "
                " WHERE run_id = $8",
                run.status.value,
                run.started_at,
                run.finished_at,
                json.dumps(run.result) if run.result is not None else None,
                run.error,
                run.next_retry_at,
                json.dumps(run.metadata or {}),
                run.run_id,
            )

    async def get(self, run_id: _uuid.UUID) -> JobRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM job_runs WHERE run_id = $1", run_id)
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
        clauses: list[str] = []
        params: list[Any] = []
        if trigger_id is not None:
            clauses.append(f"trigger_id = ${len(params) + 1}")
            params.append(trigger_id)
        if status is not None:
            clauses.append(f"status = ${len(params) + 1}")
            params.append(status)
        if statuses is not None:
            clauses.append(f"status = ANY(${len(params) + 1}::text[])")
            params.append(list(statuses))
        if since is not None:
            clauses.append(f"queued_at >= ${len(params) + 1}")
            params.append(since)
        if chat_binding_chat_id is not None:
            # spec is JSONB; ``->'chat_binding'->>'chat_id'`` extracts
            # the bound chat id without requiring a sidecar column.
            clauses.append(f"spec->'chat_binding'->>'chat_id' = ${len(params) + 1}")
            params.append(chat_binding_chat_id)
        sql = "SELECT * FROM job_runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY queued_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_run(r) for r in rows]

    async def latest_attempt(self, *, trigger_id: str, signal_id: _uuid.UUID) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(attempt_number) AS n   FROM job_runs  WHERE trigger_id = $1 AND signal_id = $2",
                trigger_id,
                signal_id,
            )
        if row is None or row["n"] is None:
            return 0
        return int(row["n"])

    async def find_latest(self, *, trigger_id: str, signal_id: _uuid.UUID) -> JobRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM job_runs "
                " WHERE trigger_id = $1 AND signal_id = $2 "
                " ORDER BY attempt_number DESC LIMIT 1",
                trigger_id,
                signal_id,
            )
        return _row_to_run(row) if row is not None else None

    async def _fetch_run_by_dedupe(
        self, *, trigger_id: str, signal_id: _uuid.UUID, attempt_number: int
    ) -> JobRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM job_runs  WHERE trigger_id = $1 AND signal_id = $2    AND attempt_number = $3",
                trigger_id,
                signal_id,
                attempt_number,
            )
        return _row_to_run(row) if row is not None else None


# ── Schedule store ───────────────────────────────────────────


class PostgresScheduleStore(OrchidScheduleStore):
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert(self, record: OrchidScheduleRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO schedules "
                "(schedule_id, trigger_id, cron, interval_seconds, "
                " identity_claim, last_fire_at, next_fire_at, enabled) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8) "
                "ON CONFLICT (schedule_id) DO UPDATE SET "
                "  trigger_id = EXCLUDED.trigger_id, "
                "  cron = EXCLUDED.cron, "
                "  interval_seconds = EXCLUDED.interval_seconds, "
                "  identity_claim = EXCLUDED.identity_claim, "
                "  last_fire_at = EXCLUDED.last_fire_at, "
                "  next_fire_at = EXCLUDED.next_fire_at, "
                "  enabled = EXCLUDED.enabled",
                record.schedule_id,
                record.trigger_id,
                record.cron,
                record.interval_seconds,
                json.dumps(record.identity_claim),
                record.last_fire_at,
                record.next_fire_at,
                record.enabled,
            )

    async def get(self, schedule_id: str) -> OrchidScheduleRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM schedules WHERE schedule_id = $1",
                schedule_id,
            )
        return _row_to_schedule(row) if row is not None else None

    async def list(self) -> Iterable[OrchidScheduleRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM schedules")
        return [_row_to_schedule(r) for r in rows]

    async def set_enabled(self, schedule_id: str, *, enabled: bool) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE schedules SET enabled = $1 WHERE schedule_id = $2",
                enabled,
                schedule_id,
            )

    async def record_fire(
        self,
        schedule_id: str,
        *,
        last_fire_at: _dt.datetime,
        next_fire_at: _dt.datetime | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE schedules    SET last_fire_at = $1, next_fire_at = $2  WHERE schedule_id = $3",
                last_fire_at,
                next_fire_at,
                schedule_id,
            )


# ── Trigger store ────────────────────────────────────────────


class PostgresTriggerStore(OrchidTriggerStore):
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert_version(self, record: OrchidTriggerRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO triggers "
                "(trigger_id, version, config, deleted_at, created_at) "
                "VALUES ($1, $2, $3::jsonb, $4, $5)",
                record.trigger_id,
                record.version,
                json.dumps(record.config),
                record.deleted_at,
                record.created_at,
            )

    async def latest(self, trigger_id: str) -> OrchidTriggerRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM triggers  WHERE trigger_id = $1 AND deleted_at IS NULL  ORDER BY version DESC LIMIT 1",
                trigger_id,
            )
        return _row_to_trigger(row) if row is not None else None

    async def list_active(self) -> Iterable[OrchidTriggerRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT t.* FROM triggers t "
                "  WHERE t.deleted_at IS NULL "
                "    AND t.version = (SELECT MAX(version) FROM triggers t2 "
                "                       WHERE t2.trigger_id = t.trigger_id "
                "                         AND t2.deleted_at IS NULL)"
            )
        return [_row_to_trigger(r) for r in rows]

    async def soft_delete(self, trigger_id: str, *, deleted_at: _dt.datetime) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE triggers SET deleted_at = $1  WHERE trigger_id = $2 AND deleted_at IS NULL",
                deleted_at,
                trigger_id,
            )


# ── Helpers ──────────────────────────────────────────────────


def _conn_from_tx(tx: DBTransaction | None) -> asyncpg.Connection | None:
    if isinstance(tx, _PostgresDBTransaction):
        return tx.conn
    return None


def _maybe_load_jsonb(value: Any) -> Any:
    """asyncpg returns JSONB columns as Python objects when a codec is
    registered, or as raw strings otherwise.  This helper is tolerant
    of both — every reader passes through it."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_signal(row: asyncpg.Record) -> Signal:
    return Signal(
        type=row["type"],
        payload=_maybe_load_jsonb(row["payload"]) or {},
        source=row["source"],
        occurred_at=row["occurred_at"],
        tenant_key=row["tenant_key"],
        signal_id=row["signal_id"],
        persisted_at=row["persisted_at"],
        user_id=row["user_id"],
        correlation_id=row["correlation_id"],
        dedupe_key=row["dedupe_key"],
        identity_claim=_maybe_load_jsonb(row["identity_claim"]),
        chat_binding=_maybe_load_jsonb(row["chat_binding"]) if "chat_binding" in row.keys() else None,
        relay_status=row["relay_status"],
    )


def _row_to_run(row: asyncpg.Record) -> JobRun:
    spec_dict = _maybe_load_jsonb(row["spec"])
    spec = JobSpec(
        trigger_id=spec_dict["trigger_id"],
        signal_id=_uuid.UUID(spec_dict["signal_id"])
        if isinstance(spec_dict["signal_id"], str)
        else spec_dict["signal_id"],
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
        run_id=row["run_id"],
        spec=spec,
        attempt_number=int(row["attempt_number"]),
        status=JobStatus(row["status"]),
        queued_at=row["queued_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        result=_maybe_load_jsonb(row["result"]),
        error=row["error"],
        next_retry_at=row["next_retry_at"],
        metadata=_maybe_load_jsonb(row["metadata"]) or {},
    )


def _row_to_schedule(row: asyncpg.Record) -> OrchidScheduleRecord:
    return OrchidScheduleRecord(
        schedule_id=row["schedule_id"],
        trigger_id=row["trigger_id"],
        cron=row["cron"],
        interval_seconds=row["interval_seconds"],
        identity_claim=_maybe_load_jsonb(row["identity_claim"]) or {},
        last_fire_at=row["last_fire_at"],
        next_fire_at=row["next_fire_at"],
        enabled=bool(row["enabled"]),
    )


def _row_to_trigger(row: asyncpg.Record) -> OrchidTriggerRecord:
    return OrchidTriggerRecord(
        trigger_id=row["trigger_id"],
        version=int(row["version"]),
        config=_maybe_load_jsonb(row["config"]) or {},
        created_at=row["created_at"],
        deleted_at=row["deleted_at"],
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
