"""Persistence ABCs for signals, jobs, schedules, and the trigger
config row.

Concrete stores live in ``orchid_ai/events/backends/`` (Postgres,
SQLite) and ``orchid_ai/events/queues/inmemory.py`` (the in-memory
reference used by tests).  All four stores typically share a single
backing connection / pool — the split exists so callers depend only on
the surface they need (interface segregation).
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Sequence

from .job import JobRun
from .queue import DBTransaction
from .signal import Signal


class OrchidSignalStore(ABC):
    """Append-mostly persistent log of signals.  Reads are by id or
    by ``(source, dedupe_key)``; updates are limited to flipping
    ``relay_status`` for the external-bus relay path."""

    @abstractmethod
    async def insert(self, signal: Signal, *, tx: DBTransaction | None = None) -> Signal:
        """Insert a signal row.  Raises :class:`SignalDuplicateError`
        when ``(source, dedupe_key)`` already exists."""

    @abstractmethod
    async def get(self, signal_id: _uuid.UUID) -> Signal | None: ...

    @abstractmethod
    async def find_by_dedupe(self, *, source: str, dedupe_key: str | None) -> _uuid.UUID | None:
        """Return the existing ``signal_id`` for the given dedupe pair,
        or ``None`` when nothing is stored.  Used by the dispatcher
        after a dedup collision to surface the original id."""

    @abstractmethod
    async def list(
        self,
        *,
        type: str | None = None,
        tenant_key: str | None = None,
        since: _dt.datetime | None = None,
        limit: int = 100,
    ) -> list[Signal]: ...

    @abstractmethod
    async def update_relay_status(self, signal_id: _uuid.UUID, *, status: str) -> None: ...


class OrchidJobStore(ABC):
    """Persistent log of job runs."""

    @abstractmethod
    async def insert(self, run: JobRun) -> JobRun:
        """Insert a new run.  The unique constraint on
        ``(trigger_id, signal_id, attempt_number)`` guarantees
        idempotency across restarts."""

    @abstractmethod
    async def update(self, run: JobRun) -> None:
        """Persist a status / result / error transition.  No version
        check — the processor's per-key serialisation already prevents
        concurrent updates."""

    @abstractmethod
    async def get(self, run_id: _uuid.UUID) -> JobRun | None: ...

    @abstractmethod
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
        """List runs, optionally filtered.

        ``status`` filters to a single status (existing behaviour).
        ``statuses`` filters to ANY of a list — used by the new
        ``GET /chats/{chat_id}/events/stream`` discovery query
        which needs ``[PENDING, RUNNING]``.  Both at once is allowed
        but redundant; the implementations AND-them.

        ``chat_binding_chat_id`` returns only runs whose
        ``spec.chat_binding.chat_id`` matches.  Added for the chat
        events stream's discovery step (Phase 4.5 §LS6); narrow,
        additive — no impact on callers that don't pass it.
        """
        ...

    @abstractmethod
    async def latest_attempt(self, *, trigger_id: str, signal_id: _uuid.UUID) -> int:
        """Return the highest ``attempt_number`` for the
        ``(trigger_id, signal_id)`` pair, or ``0`` when no run exists.
        Used by the processor when scheduling a retry."""

    @abstractmethod
    async def find_latest(self, *, trigger_id: str, signal_id: _uuid.UUID) -> JobRun | None:
        """Return the most recent ``JobRun`` for the
        ``(trigger_id, signal_id)`` pair, or ``None`` when no run
        exists.  Used by the processor to short-circuit on redelivery
        when the previous attempt is in a terminal status."""


@dataclass(frozen=True, slots=True)
class OrchidScheduleRecord:
    """Persisted schedule row.  Mirrors the SQL columns 1:1."""

    schedule_id: str
    trigger_id: str
    cron: str | None
    interval_seconds: int | None
    identity_claim: dict[str, Any]
    last_fire_at: _dt.datetime | None
    next_fire_at: _dt.datetime | None
    enabled: bool


class OrchidScheduleStore(ABC):
    @abstractmethod
    async def upsert(self, record: OrchidScheduleRecord) -> None: ...

    @abstractmethod
    async def get(self, schedule_id: str) -> OrchidScheduleRecord | None: ...

    @abstractmethod
    async def list(self) -> Iterable[OrchidScheduleRecord]: ...

    @abstractmethod
    async def set_enabled(self, schedule_id: str, *, enabled: bool) -> None: ...

    @abstractmethod
    async def record_fire(
        self,
        schedule_id: str,
        *,
        last_fire_at: _dt.datetime,
        next_fire_at: _dt.datetime | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class OrchidTriggerRecord:
    """Versioned snapshot of a trigger's YAML config.  Stored so that a
    retry firing weeks after the original signal can replay against
    the same trigger definition even if the live config has moved on."""

    trigger_id: str
    version: int
    config: dict[str, Any]
    created_at: _dt.datetime
    deleted_at: _dt.datetime | None


class OrchidTriggerStore(ABC):
    @abstractmethod
    async def insert_version(self, record: OrchidTriggerRecord) -> None: ...

    @abstractmethod
    async def latest(self, trigger_id: str) -> OrchidTriggerRecord | None: ...

    @abstractmethod
    async def list_active(self) -> Iterable[OrchidTriggerRecord]: ...

    @abstractmethod
    async def soft_delete(self, trigger_id: str, *, deleted_at: _dt.datetime) -> None: ...
