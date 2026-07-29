"""Relay queue — adapter to an external bus (Kafka / SQS / Redis Streams).

External buses can't participate in our DB transaction, so the relay
queue uses the **publish-then-mark** pattern:

1. Inside the dispatcher's outbox transaction, write the signal to the
   store with ``relay_status='pending_publish'``.  This is the
   transactional half — guaranteed to be on disk when the dispatcher
   returns.
2. After commit, call :class:`BusPublisher.publish(signal_id)` to push
   to the bus.
3. On success, flip ``relay_status='published'``.
4. If step 2 or 3 crashes, the row stays at ``pending_publish``.  A
   ``RelayRecoveryProducer`` (lands later) periodically scans the
   table and re-publishes — eventually consistent, at-least-once
   delivery.

This module ships **only the queue skeleton + the** :class:`BusPublisher`
**ABC**.  The recovery producer is intentionally deferred to the
producer phase; what's here is enough for an integrator to wire a
Kafka adapter today and rely on app-level retry / DLQ semantics.

Note that ``RelayingSignalQueue`` does NOT consume from the bus.
Consuming is the job of a separate :class:`OrchidSignalProducer`
that subscribes to the bus and calls ``dispatcher.ingest`` for each
inbound message.  This queue exists for the publish side only —
exposing in-process work to remote workers.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid as _uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ...core.events.queue import (
    DBTransaction,
    OrchidSignalQueue,
    QueuedSignal,
)
from ...core.events.store import OrchidSignalStore

_logger = logging.getLogger(__name__)


# ── BusPublisher ABC ────────────────────────────────────────


class BusPublisher(ABC):
    """A thin publish-only interface over an external bus."""

    @abstractmethod
    async def publish(self, signal_id: _uuid.UUID, *, payload_hint: dict | None = None) -> None:
        """Push one signal id (plus optional metadata) onto the bus.

        The hint is advisory — the canonical signal lives in
        :class:`OrchidSignalStore`.  Most adapters will encode just
        ``str(signal_id)`` and an opaque envelope; latency-sensitive
        adapters (Redis Streams) might inline ``payload_hint`` to
        skip a DB read on the consumer side.
        """


# ── RelayingSignalQueue ─────────────────────────────────────


class RelayingSignalQueue(OrchidSignalQueue):
    """Queue adapter that publishes each enqueue onto an external bus.

    Method semantics:

    - :meth:`enqueue` — flips ``relay_status`` to ``pending_publish``
      inside the supplied transaction; on commit, publishes to the
      bus and flips ``relay_status`` to ``published`` on success.
    - :meth:`dequeue`, :meth:`ack`, :meth:`nack`, :meth:`dead_letter`
      — **not implemented**.  A relay queue is for fan-out; consumer
      acks belong to whatever subscriber is reading the bus.  Calling
      these raises :class:`NotImplementedError` so wiring mistakes
      surface immediately rather than as silent drops.
    - :meth:`transaction` — yields the inner queue's transaction so
      the dispatcher's outbox semantics hold for the durable signal
      write.

    Construction takes an inner queue (typically the SQLite or
    Postgres queue) **and** the publisher.  The inner queue is what
    ``transaction()`` delegates to — the relay queue itself doesn't
    own a DB transaction; it just needs the signal store updates to
    commit alongside the producer's other writes.
    """

    def __init__(
        self,
        *,
        inner: OrchidSignalQueue,
        store: OrchidSignalStore,
        publisher: BusPublisher,
    ) -> None:
        self._inner = inner
        self._store = store
        self._publisher = publisher

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DBTransaction | None]:
        async with self._inner.transaction() as tx:
            yield tx

    async def enqueue(
        self,
        signal_id: _uuid.UUID,
        *,
        priority: int = 0,
        tx: DBTransaction | None = None,
    ) -> str:
        # Stage 1: write the durable side (signals.relay_status).
        # The store update happens on the same connection as the
        # dispatcher's outbox so it commits atomically.  We use the
        # sentinel queue_msg_id "relay:<signal_id>" as the visible
        # handle — there is no real signal_queue row for relay
        # messages, the bus replaces it.
        msg_id = f"relay:{signal_id}"
        await self._store.update_relay_status(signal_id, status="pending_publish")
        # Stage 2: publish.  We do this OUTSIDE the transactional
        # context — the dispatcher commits its outbox first and then
        # returns; the publish happens here, post-return.  If the
        # publisher crashes, the row stays at ``pending_publish`` for
        # the recovery producer.
        try:
            await self._publisher.publish(signal_id)
        except Exception:
            _logger.exception(
                "RelayingSignalQueue: publisher failed for %s — leaving pending_publish for recovery sweeper",
                signal_id,
            )
            # Re-raise so the caller knows publish wasn't acked; the
            # recovery sweeper will retry later.
            raise
        # Stage 3: mark published.  A crash between publish and this
        # update produces an at-least-once delivery on the bus, which
        # is the standard semantics consumers expect.
        await self._store.update_relay_status(signal_id, status="published")
        return msg_id

    async def dequeue(self, *, batch_size: int, lease_seconds: int) -> list[QueuedSignal]:
        raise NotImplementedError(
            "RelayingSignalQueue is fan-out only; consumers read from the external bus, not from this queue"
        )

    async def ack(self, queue_msg_id: str) -> None:
        raise NotImplementedError("RelayingSignalQueue.ack is not meaningful; ack happens on the external bus")

    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        raise NotImplementedError("RelayingSignalQueue.nack is not meaningful; redelivery happens on the external bus")

    async def dead_letter(self, queue_msg_id: str, *, reason: str) -> None:
        raise NotImplementedError(
            "RelayingSignalQueue.dead_letter is not meaningful; the external bus owns DLQ semantics"
        )


# ── Test stub ───────────────────────────────────────────────


class InMemoryBusPublisher(BusPublisher):
    """Reference :class:`BusPublisher` for tests + demos.

    Records every published signal id in :attr:`published`.  Ships in
    the framework rather than the test fixtures because it's also
    useful in :file:`examples/` for wiring a fake-bus demo without
    pulling in Kafka.
    """

    def __init__(self, *, fail_next: int = 0) -> None:
        self.published: list[_uuid.UUID] = []
        self._fail_next = fail_next
        self._failed_at: _dt.datetime | None = None

    async def publish(self, signal_id: _uuid.UUID, *, payload_hint: dict | None = None) -> None:
        if self._fail_next > 0:
            self._fail_next -= 1
            self._failed_at = _dt.datetime.now(tz=_dt.UTC)
            raise RuntimeError("InMemoryBusPublisher: simulated failure")
        self.published.append(signal_id)
