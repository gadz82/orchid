"""Tests for :class:`APSchedulerBackend` — the in-process clock that
the :class:`SchedulerProducer` drives.

We don't try to verify APScheduler's own correctness; the contract
we care about is just our wrapper:

- ``start()`` boots the scheduler.
- ``add_interval`` / ``add_cron`` register jobs we can spy on.
- ``remove`` is idempotent.
- ``shutdown`` is idempotent and drains in-flight fires.
"""

from __future__ import annotations

import asyncio
import datetime as _dt

import pytest

pytest.importorskip("apscheduler")

from orchid_ai.events.schedulers.apscheduler import APSchedulerBackend


@pytest.fixture
async def backend():
    b = APSchedulerBackend()
    yield b
    await b.shutdown(wait=False)


async def test_start_is_idempotent(backend: APSchedulerBackend) -> None:
    await backend.start()
    assert backend.is_running
    await backend.start()  # no-op
    assert backend.is_running


async def test_remove_is_idempotent(backend: APSchedulerBackend) -> None:
    backend.remove("never-registered")  # must not raise


async def test_interval_job_fires(backend: APSchedulerBackend) -> None:
    fires: list[_dt.datetime] = []

    async def _cb() -> None:
        fires.append(_dt.datetime.now(tz=_dt.UTC))

    await backend.start()
    # Pin the next fire time so the test doesn't wait the full
    # interval.  APScheduler treats next_run_time as an explicit
    # one-shot; subsequent fires follow the interval.
    backend.add_interval(
        schedule_id="test-interval",
        seconds=60,
        callback=_cb,
        next_run_time=_dt.datetime.now(tz=_dt.UTC) + _dt.timedelta(milliseconds=50),
    )

    # Wait up to 2s for the fire — generous to accommodate slow CI.
    for _ in range(40):
        if fires:
            break
        await asyncio.sleep(0.05)

    assert len(fires) >= 1


async def test_cron_job_registers_with_replace_existing(backend: APSchedulerBackend) -> None:
    """``add_cron`` must replace an existing job with the same id rather
    than raising — the producer relies on this when re-syncing."""
    fires: list[int] = []

    async def _cb() -> None:
        fires.append(1)

    await backend.start()
    backend.add_cron(schedule_id="dup", cron="0 0 * * *", callback=_cb)
    # Same id again — must not raise.
    backend.add_cron(schedule_id="dup", cron="0 0 * * *", callback=_cb)
    next_fire = backend.get_next_fire("dup")
    assert next_fire is not None


async def test_shutdown_is_idempotent(backend: APSchedulerBackend) -> None:
    await backend.start()
    await backend.shutdown(wait=False)
    assert not backend.is_running
    await backend.shutdown(wait=False)  # no-op


async def test_get_next_fire_returns_none_for_unknown(backend: APSchedulerBackend) -> None:
    assert backend.get_next_fire("nothing") is None
