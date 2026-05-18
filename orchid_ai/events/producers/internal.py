"""In-process bridge between code that wants to emit a signal and the
dispatcher.

Two roles:

- :class:`DispatcherSignalEmitter` — concrete
  :class:`OrchidSignalEmitter` that forwards :meth:`emit` to
  :meth:`OrchidSignalDispatcher.ingest`.  This is what
  ``AppContext.signal_emitter`` exposes.
- :class:`InternalEmissionProducer` — a thin lifecycle-managed wrapper
  so YAML can enable internal emissions through the same producer list
  used for external sources.
"""

from __future__ import annotations

from ...core.events.dispatcher import OrchidSignalDispatcher
from ...core.events.emitter import OrchidSignalEmitter
from ...core.events.producer import OrchidSignalProducer
from ...core.events.signal import SignalEnvelope, SignalIngestResult


class DispatcherSignalEmitter(OrchidSignalEmitter):
    """Thin facade — every emit becomes a dispatcher ingest call."""

    def __init__(self, dispatcher: OrchidSignalDispatcher) -> None:
        self._dispatcher = dispatcher

    async def emit(self, envelope: SignalEnvelope) -> SignalIngestResult:
        return await self._dispatcher.ingest(envelope)


class InternalEmissionProducer(OrchidSignalProducer):
    """Lifecycle wrapper for dispatcher-backed internal emissions.

    The events runtime exposes its own ``DispatcherSignalEmitter`` for
    agents and API code.  This producer keeps the documented YAML
    component path importable and startable, with no background task.
    """

    def __init__(self) -> None:
        self._emitter: DispatcherSignalEmitter | None = None

    @property
    def emitter(self) -> DispatcherSignalEmitter | None:
        return self._emitter

    async def start(self, dispatcher: OrchidSignalDispatcher) -> None:
        self._emitter = DispatcherSignalEmitter(dispatcher)

    async def stop(self) -> None:
        self._emitter = None
