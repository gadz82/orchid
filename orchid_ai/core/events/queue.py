"""Durable buffer between ingest and processing — the ABC and the
``QueuedSignal`` value object.

Concrete backends live under ``orchid_ai/events/queues/``:

- ``inmemory.py`` — reference, for tests + single-process demos.
- ``sqlite.py`` / ``postgres.py`` — production-shaped.  Both implement
  the transactional outbox via ``transaction()`` returning a
  :class:`DBTransaction` handle the dispatcher can pass back into
  :meth:`enqueue` so the ``Signal`` insert and the ``signal_queue``
  insert commit atomically.
- ``relay.py`` — adapter that wraps an external bus (Kafka / SQS /
  Redis Streams).  External buses can't participate in DB transactions,
  so the relay queue uses the publish-then-mark fallback (write
  ``signals.relay_status='pending_publish'`` first, publish next, mark
  ``'published'`` after).

The ABC is deliberately small — a queue is just a ring buffer with
leases.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QueuedSignal:
    """A leased signal pulled from the queue.

    ``lease_until`` is when the queue will redeliver this message if
    the worker hasn't called :meth:`OrchidSignalQueue.ack` or
    :meth:`OrchidSignalQueue.nack` first — crash-safety contract.
    """

    queue_msg_id: str
    signal_id: _uuid.UUID
    enqueued_at: _dt.datetime
    lease_until: _dt.datetime
    attempt: int
    payload_hint: dict[str, Any] | None = None


class DBTransaction(ABC):
    """Opaque transaction handle threaded from the queue back to the
    dispatcher.  Concrete subclasses wrap a ``psycopg`` /
    ``aiosqlite`` connection; external-bus implementations return a
    no-op handle.

    The dispatcher only needs to *pass* a ``DBTransaction``, never to
    inspect it — so the ABC stays empty."""


class OrchidSignalQueue(ABC):
    """Pluggable durable buffer between dispatcher and processor."""

    @abstractmethod
    async def enqueue(
        self,
        signal_id: _uuid.UUID,
        *,
        priority: int = 0,
        tx: DBTransaction | None = None,
    ) -> str:
        """Add a signal to the queue.  Returns ``queue_msg_id``.

        ``tx`` is the dispatcher's open store transaction; transactional
        backends commit the queue insert in the same transaction so the
        outbox stays consistent.  External-bus backends ignore it.
        """

    @abstractmethod
    async def dequeue(self, *, batch_size: int, lease_seconds: int) -> list[QueuedSignal]:
        """Atomically lease up to ``batch_size`` signals.  Empty list
        means 'queue is empty or everything visible is leased'.

        The contract is at-least-once: a leased message that is not
        ``ack``'d before ``lease_until`` becomes visible again."""

    @abstractmethod
    async def ack(self, queue_msg_id: str) -> None:
        """Mark a message processed.  Idempotent — acking an already-
        acked message must be a no-op rather than an error."""

    @abstractmethod
    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        """Return a message to the queue with a backoff.  When the
        message's ``attempt`` count is at or above ``max_attempts`` the
        backend MUST move it to the dead-letter table instead."""

    @abstractmethod
    async def dead_letter(self, queue_msg_id: str, *, reason: str) -> None:
        """Move a message to the dead-letter table.  Called explicitly
        by the processor for non-retryable failures, or implicitly by
        the queue itself once attempts are exhausted."""

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DBTransaction | None]:
        """Open a queue-side transaction.

        The default implementation yields ``None`` — backends that
        can't participate in DB transactions (Kafka, SQS) inherit this
        and the dispatcher falls back to the relay path (write signal,
        commit, then enqueue without atomicity).  Transactional
        backends override to yield a real handle.

        Implemented as a method (not a property) so subclasses can
        override with their own ``@asynccontextmanager`` decorator.
        """
        yield None
