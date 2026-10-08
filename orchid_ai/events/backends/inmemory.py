"""In-memory events storage facade — the framework default.

Composes the four in-memory stores already shipped in
:mod:`orchid_ai.events.queues.inmemory` (signals / jobs / schedules /
triggers) behind the same public surface as the durable backends:
``init_db()`` / ``close()`` plus ``signals`` / ``jobs`` / ``schedules`` /
``triggers`` accessors.

Used by :func:`orchid_ai.events.bootstrap.start_events` when
``events.enabled: true`` but no ``events.store`` is configured.  State is
process-local and **not** durable across restarts — configure a storage
plugin for persistence.
"""

from __future__ import annotations

from ..queues.inmemory import (
    InMemoryJobStore,
    InMemoryScheduleStore,
    InMemorySignalStore,
    InMemoryTriggerStore,
)

__all__ = ["InMemoryEventStorage"]


class InMemoryEventStorage:
    """Facade owning the four in-memory event stores.

    Accepts and ignores arbitrary keyword arguments so it can be built
    from an ``events.store.extra_args`` block without special-casing.
    """

    def __init__(self, **_: object) -> None:
        self._signals = InMemorySignalStore()
        self._jobs = InMemoryJobStore()
        self._schedules = InMemoryScheduleStore()
        self._triggers = InMemoryTriggerStore()

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    @property
    def signals(self) -> InMemorySignalStore:
        return self._signals

    @property
    def jobs(self) -> InMemoryJobStore:
        return self._jobs

    @property
    def schedules(self) -> InMemoryScheduleStore:
        return self._schedules

    @property
    def triggers(self) -> InMemoryTriggerStore:
        return self._triggers
