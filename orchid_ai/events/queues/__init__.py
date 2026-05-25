"""Concrete signal-queue implementations.

In-memory, SQLite, and a relay-bus skeleton.  The in-memory queue
keeps the in-memory stores (signals, jobs, schedules, triggers)
under one roof so unit tests can drive the dispatcher and processor
without spinning up a database; the durable queues pair with the
matching event-store class in ``events/backends/``.

The PostgreSQL queue lives in ``orchid-storage-postgres``.
"""

from __future__ import annotations

from .inmemory import (
    InMemoryJobStore,
    InMemoryScheduleStore,
    InMemorySignalQueue,
    InMemorySignalStore,
    InMemoryTriggerStore,
)
from .relay import BusPublisher, InMemoryBusPublisher, RelayingSignalQueue

__all__ = [
    "BusPublisher",
    "InMemoryBusPublisher",
    "InMemoryJobStore",
    "InMemoryScheduleStore",
    "InMemorySignalQueue",
    "InMemorySignalStore",
    "InMemoryTriggerStore",
    "RelayingSignalQueue",
    "SQLiteSignalQueue",
]


def __getattr__(name: str) -> object:
    if name == "SQLiteSignalQueue":
        from .sqlite import SQLiteSignalQueue

        return SQLiteSignalQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
