"""Backends for the events stores.

:class:`InMemoryEventStorage` is the dependency-free framework default —
used when ``events.enabled: true`` but no ``events.store`` is configured.
It composes the four in-memory stores from
:mod:`orchid_ai.events.queues.inmemory`.

Durable backends live in plugin packages:

- SQLite — ``orchid-storage-sqlite``
  (``orchid_storage_sqlite.event_storage.SQLiteEventStorage``).
- PostgreSQL — ``orchid-storage-postgres``
  (``orchid_storage_postgres.event_storage.PostgresEventStorage``).
"""

from __future__ import annotations

from .inmemory import InMemoryEventStorage

__all__ = ["InMemoryEventStorage"]
