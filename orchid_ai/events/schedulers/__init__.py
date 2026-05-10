"""Scheduler backends.

A "scheduler backend" is the in-process clock that fires cron /
interval timers.  Today we ship one — :class:`APSchedulerBackend`
wrapping :mod:`apscheduler` — but the contract (set on
:class:`OrchidSchedulerBackend` in :mod:`...core.events.scheduler`
when that lands in a later phase) is intentionally narrow so an
integrator can swap in their own (e.g. a distributed scheduler) by
pointing YAML at a different dotted path.

The scheduler backend is *not* the durable source of truth for
schedules — that's :class:`OrchidScheduleStore`.  The backend reads
schedules from the store at start-up and re-registers them on every
process boot, which is what gives Pollen + Bloom its restart
durability without a SQLAlchemy dependency.
"""

from __future__ import annotations
