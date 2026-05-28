"""APScheduler-backed clock for cron / interval triggers.

This is the in-process timer that the :class:`SchedulerProducer`
drives.  It wraps :class:`apscheduler.schedulers.asyncio.AsyncIOScheduler`
with a tiny lifecycle facade so the producer can:

- ``start()`` — boot the scheduler.
- ``add_cron(schedule_id, cron, callback)`` — register a cron job.
- ``add_interval(schedule_id, seconds, callback)`` — register an
  interval job.
- ``remove(schedule_id)`` — cancel a registered job.
- ``shutdown()`` — drain in-flight jobs and stop.

Why we don't use APScheduler's persistent jobstores:

The spec says the ``schedules`` table is the source of truth.
APScheduler's stock SQLAlchemyJobStore would require us to take a
top-level dependency on SQLAlchemy and serialise callable refs.
Instead we keep APScheduler in its default (in-memory) jobstore and
re-register jobs on every boot from our own :class:`OrchidScheduleStore`.
That trade-off costs us a few microseconds at startup and saves us
SQLAlchemy.

``last_fire_at`` / ``next_fire_at`` on the schedule record are
maintained by the :class:`SchedulerProducer` after each fire, so
operators have an accurate view via ``GET /schedules`` regardless of
APScheduler's in-memory state.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Awaitable, Callable

_logger = logging.getLogger(__name__)


# Lazy-import APScheduler so projects that don't enable events don't
# pay the import cost.  The exception text is friendly because this
# is the most common "wait what dependency?" stumble.
#
# Raises :class:`ImportError` (not :class:`RuntimeError`) so callers
# can ``except ImportError`` uniformly across the parsers + scheduler
# extras gates — see ``documents/parsers.py`` for the same pattern.
def _apscheduler_imports() -> tuple[Any, Any, Any]:
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError as exc:  # pragma: no cover — exercised only when dep missing
        raise ImportError(
            "APSchedulerBackend requires the 'apscheduler' package, "
            "which is not installed.\n"
            "Install via `pip install orchid-ai[events]` to enable "
            "event-driven activation (Pollen + Bloom), or install "
            "`apscheduler` directly."
        ) from exc
    return AsyncIOScheduler, CronTrigger, IntervalTrigger


class APSchedulerBackend:
    """Thin wrapper around AsyncIOScheduler."""

    def __init__(self, *, timezone: str = "UTC") -> None:
        AsyncIOScheduler, _, _ = _apscheduler_imports()
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._timezone = timezone

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Boot the scheduler.  Idempotent — calling on an already-
        running scheduler is a no-op."""
        if not self._scheduler.running:
            self._scheduler.start()

    async def shutdown(self, *, wait: bool = True) -> None:
        """Stop the scheduler.  Drains in-flight jobs by default.

        ``AsyncIOScheduler.shutdown`` defers the real teardown to the
        next event-loop tick (it wraps ``BaseScheduler.shutdown`` in a
        ``call_soon_threadsafe``).  We bypass that indirection by
        invoking ``BaseScheduler.shutdown`` directly — the call is
        already running on the event loop, and we want callers to be
        able to ``await shutdown()`` and reason about ``is_running``
        synchronously immediately afterwards.
        """
        if not self._scheduler.running:
            return
        # Use the unbound parent method so the synchronous transition
        # to ``STATE_STOPPED`` happens before this coroutine resumes.
        from apscheduler.schedulers.base import BaseScheduler

        BaseScheduler.shutdown(self._scheduler, wait=wait)
        # Cancel the AsyncIOScheduler-specific timer if it's still set.
        try:
            self._scheduler._stop_timer()  # noqa: SLF001
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        return bool(self._scheduler.running)

    # ── Job registration ─────────────────────────────────────

    def add_cron(
        self,
        *,
        schedule_id: str,
        cron: str,
        callback: Callable[[], Awaitable[None]],
        max_instances: int = 1,
    ) -> None:
        """Register a cron job.  ``cron`` is a standard 5- or 6-field
        crontab expression.  The callback is async; APScheduler will
        await it on each fire."""
        _, CronTrigger, _ = _apscheduler_imports()
        trigger = CronTrigger.from_crontab(cron, timezone=self._timezone)
        self._scheduler.add_job(
            callback,
            trigger=trigger,
            id=schedule_id,
            replace_existing=True,
            max_instances=max_instances,
            coalesce=True,
            misfire_grace_time=60,
        )

    def add_interval(
        self,
        *,
        schedule_id: str,
        seconds: int,
        callback: Callable[[], Awaitable[None]],
        max_instances: int = 1,
        next_run_time: _dt.datetime | None = None,
    ) -> None:
        """Register an interval job.  ``seconds`` is the gap between
        fires.  ``next_run_time`` lets tests pin the first fire so
        they don't have to wait the full interval."""
        _, _, IntervalTrigger = _apscheduler_imports()
        trigger = IntervalTrigger(seconds=seconds, timezone=self._timezone)
        kwargs: dict[str, Any] = {}
        if next_run_time is not None:
            kwargs["next_run_time"] = next_run_time
        self._scheduler.add_job(
            callback,
            trigger=trigger,
            id=schedule_id,
            replace_existing=True,
            max_instances=max_instances,
            coalesce=True,
            misfire_grace_time=60,
            **kwargs,
        )

    def remove(self, schedule_id: str) -> None:
        """Cancel a registered job.  Silent when the schedule isn't
        registered — the producer can call this defensively on every
        ``set_enabled(False)`` without a pre-check."""
        try:
            self._scheduler.remove_job(schedule_id)
        except Exception:
            # APScheduler raises ``JobLookupError`` (subclass of
            # ``KeyError``) when the job doesn't exist.  Swallow —
            # idempotent removal is the contract.
            pass

    def get_next_fire(self, schedule_id: str) -> _dt.datetime | None:
        """Return the next scheduled fire time for the given job, or
        None if the job is unregistered.  Used by the producer to
        keep ``schedules.next_fire_at`` accurate."""
        try:
            job = self._scheduler.get_job(schedule_id)
        except Exception:
            return None
        if job is None:
            return None
        return job.next_run_time
