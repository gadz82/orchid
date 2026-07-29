"""In-memory queue + stores.

These are the reference backends used by the framework's own unit
tests and by the demo ``examples/`` projects that don't want a
database.  They are deliberately simple — single-process, single-host,
``asyncio.Lock``-protected dicts — and *not* intended for production
use.

All four ABCs (``OrchidSignalQueue``, ``OrchidSignalStore``,
``OrchidJobStore``, ``OrchidScheduleStore``, ``OrchidTriggerStore``)
share a single backing object so a test can construct one ``Bundle``
and pass it everywhere a store is needed.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager

from ...core.events.errors import SignalDuplicateError
from ...core.events.job import JobRun
from ...core.events.queue import DBTransaction, OrchidSignalQueue, QueuedSignal
from ...core.events.signal import Signal
from ...core.events.store import (
    OrchidJobStore,
    OrchidScheduleRecord,
    OrchidScheduleStore,
    OrchidSignalStore,
    OrchidTriggerRecord,
    OrchidTriggerStore,
)


class _InMemoryTransaction(DBTransaction):
    """No-op transaction handle.  In-memory stores don't actually need
    transactional grouping — the dispatcher just needs *some* handle
    to thread through ``enqueue``, and 'no-op' is the simplest
    contract."""


class InMemorySignalStore(OrchidSignalStore):
    """Process-local signal log."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._signals: dict[_uuid.UUID, Signal] = {}
        self._dedupe_index: dict[tuple[str, str | None], _uuid.UUID] = {}

    async def insert(self, signal: Signal, *, tx: DBTransaction | None = None) -> Signal:
        async with self._lock:
            key = (signal.source, signal.dedupe_key)
            if signal.dedupe_key is not None and key in self._dedupe_index:
                raise SignalDuplicateError(str(self._dedupe_index[key]))
            self._signals[signal.signal_id] = signal
            if signal.dedupe_key is not None:
                self._dedupe_index[key] = signal.signal_id
            return signal

    async def get(self, signal_id: _uuid.UUID) -> Signal | None:
        return self._signals.get(signal_id)

    async def find_by_dedupe(self, *, source: str, dedupe_key: str | None) -> _uuid.UUID | None:
        if dedupe_key is None:
            return None
        return self._dedupe_index.get((source, dedupe_key))

    async def list(
        self,
        *,
        type: str | None = None,
        tenant_key: str | None = None,
        since: _dt.datetime | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        out: list[Signal] = []
        for sig in sorted(self._signals.values(), key=lambda s: s.persisted_at, reverse=True):
            if type is not None and sig.type != type:
                continue
            if tenant_key is not None and sig.tenant_key != tenant_key:
                continue
            if since is not None and sig.persisted_at < since:
                continue
            out.append(sig)
            if len(out) >= limit:
                break
        return out

    async def list_by_relay_status(self, *, status: str, limit: int = 100) -> list[Signal]:
        """Pull signals by ``relay_status``.

        Used by :class:`RelayRecoveryProducer`'s preferred-path
        accessor; falling back to a generic ``list`` plus filtering
        works but is O(n) over the whole store.
        """
        out: list[Signal] = []
        for sig in sorted(self._signals.values(), key=lambda s: s.persisted_at):
            if sig.relay_status != status:
                continue
            out.append(sig)
            if len(out) >= limit:
                break
        return out

    async def update_relay_status(self, signal_id: _uuid.UUID, *, status: str) -> None:
        async with self._lock:
            sig = self._signals.get(signal_id)
            if sig is None:
                return
            # frozen dataclass — rebuild with the new status
            self._signals[signal_id] = Signal(
                type=sig.type,
                payload=sig.payload,
                source=sig.source,
                occurred_at=sig.occurred_at,
                tenant_key=sig.tenant_key,
                signal_id=sig.signal_id,
                persisted_at=sig.persisted_at,
                user_id=sig.user_id,
                correlation_id=sig.correlation_id,
                dedupe_key=sig.dedupe_key,
                identity_claim=sig.identity_claim,
                chat_binding=sig.chat_binding,
                relay_status=status,
            )


class InMemorySignalQueue(OrchidSignalQueue):
    """Single-process queue with leases and a dead-letter list."""

    def __init__(self, *, max_attempts: int = 5) -> None:
        self._lock = asyncio.Lock()
        self._messages: dict[str, _MutableQueuedSignal] = {}
        self._dead: dict[str, _DeadLetter] = {}
        self._max_attempts = max_attempts
        self._sequence = 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DBTransaction | None]:
        # No-op for the in-memory backend; the dispatcher still calls
        # ``insert`` and ``enqueue`` in order, which is enough for
        # tests.
        yield _InMemoryTransaction()

    async def enqueue(
        self,
        signal_id: _uuid.UUID,
        *,
        priority: int = 0,
        tx: DBTransaction | None = None,
    ) -> str:
        async with self._lock:
            self._sequence += 1
            msg_id = f"q-{self._sequence:06d}"
            now = _now()
            self._messages[msg_id] = _MutableQueuedSignal(
                queue_msg_id=msg_id,
                signal_id=signal_id,
                priority=priority,
                enqueued_at=now,
                visible_after=now,
                lease_until=None,
                attempt=0,
            )
            return msg_id

    async def dequeue(self, *, batch_size: int, lease_seconds: int) -> list[QueuedSignal]:
        out: list[QueuedSignal] = []
        async with self._lock:
            now = _now()
            # Order by priority desc, enqueued_at asc — same as the
            # SQL backend's ``ORDER BY priority DESC, enqueued_at``.
            candidates = sorted(
                (m for m in self._messages.values() if m.is_visible(now)),
                key=lambda m: (-m.priority, m.enqueued_at),
            )
            for msg in candidates[:batch_size]:
                msg.attempt += 1
                msg.lease_until = now + _dt.timedelta(seconds=lease_seconds)
                out.append(
                    QueuedSignal(
                        queue_msg_id=msg.queue_msg_id,
                        signal_id=msg.signal_id,
                        enqueued_at=msg.enqueued_at,
                        lease_until=msg.lease_until,
                        attempt=msg.attempt,
                    )
                )
        return out

    async def ack(self, queue_msg_id: str) -> None:
        async with self._lock:
            self._messages.pop(queue_msg_id, None)

    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        async with self._lock:
            msg = self._messages.get(queue_msg_id)
            if msg is None:
                return
            if msg.attempt >= self._max_attempts:
                self._move_to_dead_letter(msg, reason="max attempts exceeded")
                return
            msg.lease_until = None
            msg.visible_after = _now() + _dt.timedelta(seconds=retry_after_seconds)

    async def dead_letter(self, queue_msg_id: str, *, reason: str) -> None:
        async with self._lock:
            msg = self._messages.get(queue_msg_id)
            if msg is None:
                return
            self._move_to_dead_letter(msg, reason=reason)

    # ── helpers ──────────────────────────────────────────

    def _move_to_dead_letter(self, msg: _MutableQueuedSignal, *, reason: str) -> None:
        self._dead[msg.queue_msg_id] = _DeadLetter(
            queue_msg_id=msg.queue_msg_id,
            signal_id=msg.signal_id,
            reason=reason,
            failed_at=_now(),
            attempts=msg.attempt,
        )
        self._messages.pop(msg.queue_msg_id, None)

    @property
    def dead_letters(self) -> dict[str, _DeadLetter]:
        """Test-only view of the dead-letter list."""
        return dict(self._dead)

    @property
    def visible_messages(self) -> int:
        now = _now()
        return sum(1 for m in self._messages.values() if m.is_visible(now))

    @property
    def in_flight(self) -> int:
        now = _now()
        return sum(1 for m in self._messages.values() if not m.is_visible(now))


class _MutableQueuedSignal:
    """Mutable internal counterpart of :class:`QueuedSignal`."""

    __slots__ = (
        "attempt",
        "enqueued_at",
        "lease_until",
        "priority",
        "queue_msg_id",
        "signal_id",
        "visible_after",
    )

    def __init__(
        self,
        *,
        queue_msg_id: str,
        signal_id: _uuid.UUID,
        priority: int,
        enqueued_at: _dt.datetime,
        visible_after: _dt.datetime,
        lease_until: _dt.datetime | None,
        attempt: int,
    ) -> None:
        self.queue_msg_id = queue_msg_id
        self.signal_id = signal_id
        self.priority = priority
        self.enqueued_at = enqueued_at
        self.visible_after = visible_after
        self.lease_until = lease_until
        self.attempt = attempt

    def is_visible(self, now: _dt.datetime) -> bool:
        if self.visible_after > now:
            return False
        if self.lease_until is None:
            return True
        return self.lease_until <= now


class _DeadLetter:
    __slots__ = ("attempts", "failed_at", "queue_msg_id", "reason", "signal_id")

    def __init__(
        self,
        *,
        queue_msg_id: str,
        signal_id: _uuid.UUID,
        reason: str,
        failed_at: _dt.datetime,
        attempts: int,
    ) -> None:
        self.queue_msg_id = queue_msg_id
        self.signal_id = signal_id
        self.reason = reason
        self.failed_at = failed_at
        self.attempts = attempts


# ── Job / schedule / trigger stores ────────────────────────


class InMemoryJobStore(OrchidJobStore):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[_uuid.UUID, JobRun] = {}
        self._dedupe_index: dict[tuple[str, _uuid.UUID, int], _uuid.UUID] = {}

    async def insert(self, run: JobRun) -> JobRun:
        async with self._lock:
            key = (run.spec.trigger_id, run.spec.signal_id, run.attempt_number)
            if key in self._dedupe_index:
                # Same key already exists — return the original.  This
                # mirrors how a real DB UNIQUE constraint behaves under
                # a redelivered queue message.
                return self._runs[self._dedupe_index[key]]
            self._runs[run.run_id] = run
            self._dedupe_index[key] = run.run_id
            return run

    async def update(self, run: JobRun) -> None:
        async with self._lock:
            self._runs[run.run_id] = run

    async def get(self, run_id: _uuid.UUID) -> JobRun | None:
        return self._runs.get(run_id)

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
        statuses_set = set(statuses) if statuses is not None else None
        out: list[JobRun] = []
        for run in sorted(self._runs.values(), key=lambda r: r.queued_at, reverse=True):
            if trigger_id is not None and run.spec.trigger_id != trigger_id:
                continue
            if status is not None and run.status.value != status:
                continue
            if statuses_set is not None and run.status.value not in statuses_set:
                continue
            if since is not None and run.queued_at < since:
                continue
            if chat_binding_chat_id is not None:
                binding = run.spec.chat_binding
                if not binding or binding.get("chat_id") != chat_binding_chat_id:
                    continue
            out.append(run)
            if len(out) >= limit:
                break
        return out

    async def latest_attempt(self, *, trigger_id: str, signal_id: _uuid.UUID) -> int:
        attempts = [n for (tid, sid, n) in self._dedupe_index if tid == trigger_id and sid == signal_id]
        return max(attempts, default=0)

    async def find_latest(self, *, trigger_id: str, signal_id: _uuid.UUID) -> JobRun | None:
        candidates = [
            run for run in self._runs.values() if run.spec.trigger_id == trigger_id and run.spec.signal_id == signal_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.attempt_number)


class InMemoryScheduleStore(OrchidScheduleStore):
    def __init__(self) -> None:
        self._records: dict[str, OrchidScheduleRecord] = {}

    async def upsert(self, record: OrchidScheduleRecord) -> None:
        self._records[record.schedule_id] = record

    async def get(self, schedule_id: str) -> OrchidScheduleRecord | None:
        return self._records.get(schedule_id)

    async def list(self) -> Iterable[OrchidScheduleRecord]:
        return list(self._records.values())

    async def set_enabled(self, schedule_id: str, *, enabled: bool) -> None:
        rec = self._records.get(schedule_id)
        if rec is None:
            return
        self._records[schedule_id] = OrchidScheduleRecord(
            schedule_id=rec.schedule_id,
            trigger_id=rec.trigger_id,
            cron=rec.cron,
            interval_seconds=rec.interval_seconds,
            identity_claim=rec.identity_claim,
            last_fire_at=rec.last_fire_at,
            next_fire_at=rec.next_fire_at,
            enabled=enabled,
        )

    async def record_fire(
        self,
        schedule_id: str,
        *,
        last_fire_at: _dt.datetime,
        next_fire_at: _dt.datetime | None,
    ) -> None:
        rec = self._records.get(schedule_id)
        if rec is None:
            return
        self._records[schedule_id] = OrchidScheduleRecord(
            schedule_id=rec.schedule_id,
            trigger_id=rec.trigger_id,
            cron=rec.cron,
            interval_seconds=rec.interval_seconds,
            identity_claim=rec.identity_claim,
            last_fire_at=last_fire_at,
            next_fire_at=next_fire_at,
            enabled=rec.enabled,
        )


class InMemoryTriggerStore(OrchidTriggerStore):
    def __init__(self) -> None:
        self._versions: dict[str, list[OrchidTriggerRecord]] = {}

    async def insert_version(self, record: OrchidTriggerRecord) -> None:
        self._versions.setdefault(record.trigger_id, []).append(record)

    async def latest(self, trigger_id: str) -> OrchidTriggerRecord | None:
        versions = self._versions.get(trigger_id)
        if not versions:
            return None
        active = [v for v in versions if v.deleted_at is None]
        if not active:
            return None
        return max(active, key=lambda v: v.version)

    async def list_active(self) -> Iterable[OrchidTriggerRecord]:
        out: list[OrchidTriggerRecord] = []
        for tid in self._versions:
            latest = await self.latest(tid)
            if latest is not None:
                out.append(latest)
        return out

    async def soft_delete(self, trigger_id: str, *, deleted_at: _dt.datetime) -> None:
        versions = self._versions.get(trigger_id, [])
        for i, v in enumerate(versions):
            if v.deleted_at is None:
                versions[i] = OrchidTriggerRecord(
                    trigger_id=v.trigger_id,
                    version=v.version,
                    config=v.config,
                    created_at=v.created_at,
                    deleted_at=deleted_at,
                )


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)
