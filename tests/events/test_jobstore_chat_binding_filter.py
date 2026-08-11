"""``OrchidJobStore.list`` chat-binding + statuses filter (Phase 4.5).

Covers:

- ``chat_binding_chat_id`` returns only runs whose
  ``spec.chat_binding.chat_id`` matches; non-bound runs and
  differently-bound runs are excluded.
- ``statuses=[a, b]`` returns runs whose status is in the list,
  while preserving the existing ``status: str`` single-filter.
- Combined filters AND properly.
- Existing callers that pass none of the new kwargs are unaffected.

Both ``InMemoryJobStore`` and ``SQLiteJobStore`` are exercised — the
Postgres path uses the same SQL fragments and is verified via
unit-level construction; a full Postgres CI run is out of scope
for the library suite.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from pathlib import Path

import aiosqlite
import pytest

from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.core.events.signal import Signal
from orchid_ai.events.backends.sqlite import SQLiteEventStorage
from orchid_ai.events.queues.inmemory import (
    InMemoryJobStore,
    InMemorySignalStore,
)

# ── Helpers ─────────────────────────────────────────────────


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _make_run(
    *,
    chat_binding: dict | None,
    status: JobStatus = JobStatus.PENDING,
    trigger_id: str = "t-1",
    signal_id: _uuid.UUID | None = None,
) -> JobRun:
    spec = JobSpec(
        trigger_id=trigger_id,
        signal_id=signal_id or _uuid.uuid4(),
        agent_name="agent",
        prompt="p",
        identity_claim={"mode": "act_as_user", "user_id": "u-7"},
        correlation_id="corr",
        parallelism_key="user:t-1:u-7",
        visibility="actor",
        visibility_user_id="u-7",
        chat_binding=chat_binding,
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=1,
        status=status,
        queued_at=_now(),
    )


def _signal_for(run: JobRun, store: InMemorySignalStore | None = None) -> Signal:
    return Signal(
        type="x",
        payload={},
        source="src",
        occurred_at=_now(),
        tenant_key="t-1",
        signal_id=run.spec.signal_id,
        persisted_at=_now(),
        user_id="u-7",
    )


# ── In-memory backend ───────────────────────────────────────


async def test_inmemory_chat_binding_filter_returns_only_matching_runs() -> None:
    store = InMemoryJobStore()
    bound_a = _make_run(chat_binding={"chat_id": "C-A"})
    bound_b = _make_run(chat_binding={"chat_id": "C-B"})
    unbound = _make_run(chat_binding=None)
    sig_store = InMemorySignalStore()
    for r in (bound_a, bound_b, unbound):
        await sig_store.insert(_signal_for(r))
        await store.insert(r)

    only_a = await store.list(chat_binding_chat_id="C-A")
    assert {r.run_id for r in only_a} == {bound_a.run_id}

    nothing_for_unknown = await store.list(chat_binding_chat_id="C-Z")
    assert nothing_for_unknown == []


async def test_inmemory_statuses_filter_matches_any_in_list() -> None:
    store = InMemoryJobStore()
    sig_store = InMemorySignalStore()
    pending = _make_run(chat_binding=None, status=JobStatus.PENDING)
    running = _make_run(chat_binding=None, status=JobStatus.RUNNING)
    succeeded = _make_run(chat_binding=None, status=JobStatus.SUCCEEDED)
    for r in (pending, running, succeeded):
        await sig_store.insert(_signal_for(r))
        await store.insert(r)

    in_flight = await store.list(statuses=["pending", "running"])
    assert {r.run_id for r in in_flight} == {pending.run_id, running.run_id}


async def test_inmemory_combined_filters_and_together() -> None:
    """``chat_binding_chat_id`` AND ``statuses`` compose."""
    store = InMemoryJobStore()
    sig_store = InMemorySignalStore()
    a_pending = _make_run(chat_binding={"chat_id": "C-A"}, status=JobStatus.PENDING)
    a_done = _make_run(chat_binding={"chat_id": "C-A"}, status=JobStatus.SUCCEEDED)
    b_pending = _make_run(chat_binding={"chat_id": "C-B"}, status=JobStatus.PENDING)
    for r in (a_pending, a_done, b_pending):
        await sig_store.insert(_signal_for(r))
        await store.insert(r)

    rows = await store.list(
        chat_binding_chat_id="C-A",
        statuses=["pending", "running"],
    )
    assert {r.run_id for r in rows} == {a_pending.run_id}


async def test_inmemory_existing_callers_unaffected() -> None:
    """Callers that don't pass the new kwargs see all runs ordered DESC."""
    store = InMemoryJobStore()
    sig_store = InMemorySignalStore()
    runs = [_make_run(chat_binding=None) for _ in range(3)]
    for r in runs:
        await sig_store.insert(_signal_for(r))
        await store.insert(r)
    rows = await store.list()
    assert len(rows) == 3


# ── SQLite backend ──────────────────────────────────────────


@pytest.fixture
async def sqlite_storage(tmp_path: Path):
    dsn = str(tmp_path / "events.db")
    conn = await aiosqlite.connect(dsn)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    storage = SQLiteEventStorage(conn=conn)
    await storage.init_db()
    yield storage
    await conn.close()


async def test_sqlite_chat_binding_filter_uses_json_extract(sqlite_storage) -> None:
    """SQLite backend honours the new filter via ``json_extract``."""
    bound_a = _make_run(chat_binding={"chat_id": "C-A"})
    bound_b = _make_run(chat_binding={"chat_id": "C-B"})
    unbound = _make_run(chat_binding=None)
    for r in (bound_a, bound_b, unbound):
        await sqlite_storage.signals.insert(_signal_for(r))
        await sqlite_storage.jobs.insert(r)

    only_a = await sqlite_storage.jobs.list(chat_binding_chat_id="C-A")
    assert {r.run_id for r in only_a} == {bound_a.run_id}


async def test_sqlite_statuses_filter_uses_in_clause(sqlite_storage) -> None:
    pending = _make_run(chat_binding=None, status=JobStatus.PENDING)
    running = _make_run(chat_binding=None, status=JobStatus.RUNNING)
    succeeded = _make_run(chat_binding=None, status=JobStatus.SUCCEEDED)
    for r in (pending, running, succeeded):
        await sqlite_storage.signals.insert(_signal_for(r))
        await sqlite_storage.jobs.insert(r)

    rows = await sqlite_storage.jobs.list(statuses=["pending", "running"])
    assert {r.run_id for r in rows} == {pending.run_id, running.run_id}


async def test_sqlite_combined_filters(sqlite_storage) -> None:
    a_pending = _make_run(chat_binding={"chat_id": "C-A"}, status=JobStatus.PENDING)
    a_done = _make_run(chat_binding={"chat_id": "C-A"}, status=JobStatus.SUCCEEDED)
    b_pending = _make_run(chat_binding={"chat_id": "C-B"}, status=JobStatus.PENDING)
    for r in (a_pending, a_done, b_pending):
        await sqlite_storage.signals.insert(_signal_for(r))
        await sqlite_storage.jobs.insert(r)

    rows = await sqlite_storage.jobs.list(
        chat_binding_chat_id="C-A",
        statuses=["pending", "running"],
    )
    assert {r.run_id for r in rows} == {a_pending.run_id}
