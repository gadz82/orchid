"""Concrete signal-queue implementations.

In-memory, SQLite, Postgres, and a relay-bus skeleton.  The
in-memory queue keeps the in-memory stores (signals, jobs,
schedules, triggers) under one roof so unit tests can drive the
dispatcher and processor without spinning up a database; the
durable queues each pair with the matching event-store class in
``events/backends/``.
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
    "PostgresSignalQueue",
    "RelayingSignalQueue",
    "SQLiteSignalQueue",
]


def __getattr__(name: str) -> object:
    if name == "PostgresSignalQueue":
        from .postgres import PostgresSignalQueue

        return PostgresSignalQueue
    if name == "SQLiteSignalQueue":
        from .sqlite import SQLiteSignalQueue

        return SQLiteSignalQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
