"""Drains the queue, resolves identity, matches triggers, fires runs.

The processor is the *only* component in the events pipeline that
touches LangGraph (via the :class:`OrchidJobRunner` it is wired with).
Producers don't, the dispatcher doesn't, the queue doesn't — that
strict separation is what keeps ingest under its latency target.

Concrete implementations live under ``orchid_ai/events/processors/``:

- ``asyncio_pool.py`` — built-in default; configurable concurrency in
  the same process.
- Integrators can substitute ``CeleryProcessor``,
  ``KafkaConsumerGroupProcessor``, ``LambdaProcessor`` — any worker
  topology — by implementing this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .queue import OrchidSignalQueue
from .runner import OrchidJobRunner
from .store import OrchidJobStore, OrchidSignalStore
from .trigger import TriggerRegistry


class OrchidSignalProcessor(ABC):
    """Pluggable drain-and-dispatch worker."""

    @abstractmethod
    async def start(
        self,
        *,
        queue: OrchidSignalQueue,
        signal_store: OrchidSignalStore,
        triggers: TriggerRegistry,
        identity_resolver: Any,  # OrchidIdentityResolver — kept loose to avoid a core/identity import cycle
        job_store: OrchidJobStore,
        job_runner: OrchidJobRunner,
    ) -> None:
        """Begin draining ``queue``.  Implementations spawn worker
        tasks and return; ``start`` must not block."""

    @abstractmethod
    async def stop(self) -> None:
        """Signal workers to drain and stop.  Idempotent.  After
        ``stop`` returns, no new ``ack`` / ``nack`` calls happen against
        the queue."""
