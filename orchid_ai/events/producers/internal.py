"""In-process bridge between code that wants to emit a signal and the
dispatcher.

Two roles:

- :class:`DispatcherSignalEmitter` — concrete
  :class:`OrchidSignalEmitter` that forwards :meth:`emit` to
  :meth:`OrchidSignalDispatcher.ingest`.  This is what
  ``AppContext.signal_emitter`` exposes.
- (future, lands in the agent-binding phase) ``InternalEmissionProducer``
  — a thin lifecycle-managed wrapper that registers itself with the
  framework so the same component model applies to internal emissions
  as to external producers.  Phase 1 only needs the emitter.
"""

from __future__ import annotations

from ...core.events.dispatcher import OrchidSignalDispatcher
from ...core.events.emitter import OrchidSignalEmitter
from ...core.events.signal import SignalEnvelope, SignalIngestResult


class DispatcherSignalEmitter(OrchidSignalEmitter):
    """Thin facade — every emit becomes a dispatcher ingest call."""

    def __init__(self, dispatcher: OrchidSignalDispatcher) -> None:
        self._dispatcher = dispatcher

    async def emit(self, envelope: SignalEnvelope) -> SignalIngestResult:
        return await self._dispatcher.ingest(envelope)
