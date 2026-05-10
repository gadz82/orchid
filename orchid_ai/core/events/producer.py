"""Long-running source of signals — the ABC.

A producer is whatever turns the outside world into a
:class:`SignalEnvelope` and hands it to the dispatcher.  Built-in
producers (``HTTPIngestionProducer``, ``SchedulerProducer``,
``InternalEmissionProducer``, ``MCPIngestionProducer``) live under
``orchid_ai/events/producers/``.  Integrators add their own
(``KafkaSubscriber``, ``SQSPoller``, …) by subclassing and pointing a
YAML entry at the dotted import path.

Lifecycle: every producer is started once at framework boot via
:meth:`start`, and stopped during graceful shutdown via :meth:`stop`.
The dispatcher is passed to ``start`` so the producer can hold its own
reference rather than reaching back through any global.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .dispatcher import OrchidSignalDispatcher


class OrchidSignalProducer(ABC):
    """A long-running source of signals."""

    @property
    def name(self) -> str:
        """Stable identifier used in logs / traces.  Defaults to the
        concrete class name; subclasses can override for clarity (e.g.
        a Kafka subscriber tagging the topic name)."""
        return self.__class__.__name__

    @abstractmethod
    async def start(self, dispatcher: OrchidSignalDispatcher) -> None:
        """Start producing.  Implementations should not block — spawn
        background tasks and return."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop producing.  Must be idempotent and must drain any
        in-flight ingest calls before returning."""
