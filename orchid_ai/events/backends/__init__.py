"""Persistent backends for the events stores.

Each backend exposes four narrow store classes
(:class:`OrchidSignalStore`, :class:`OrchidJobStore`,
:class:`OrchidScheduleStore`, :class:`OrchidTriggerStore`) that share
one connection / pool with the chat-storage backend; migration
``v001`` creates the tables, the chat storage's migration runner
applies it on boot, and the event stores reuse the result.

A :class:`SQLiteEventStorage` facade owns the lifecycle (open, run
migrations, close) and exposes the four stores via attributes
(``signals``, ``jobs``, ``schedules``, ``triggers``).

The PostgreSQL backend lives in ``orchid-storage-postgres``.
"""

from __future__ import annotations

from .sqlite import (
    SQLiteEventStorage,
    SQLiteJobStore,
    SQLiteScheduleStore,
    SQLiteSignalStore,
    SQLiteTriggerStore,
)

__all__ = [
    "SQLiteEventStorage",
    "SQLiteJobStore",
    "SQLiteScheduleStore",
    "SQLiteSignalStore",
    "SQLiteTriggerStore",
]
