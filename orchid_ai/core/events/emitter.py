"""Narrow ``emit``-only interface for in-process producers.

Code that wants to emit a signal *without* depending on the full
:class:`OrchidSignalDispatcher` (concrete agents, services, internal
hooks) depends on this ABC instead.  The concrete
``DispatcherSignalEmitter`` (in
``orchid_ai/events/producers/internal.py``) is a thin facade that
forwards to ``dispatcher.ingest``.

The split exists so that ``OrchidAgent.emit_signal`` doesn't reach for
the full dispatcher API surface — interface segregation, agents can be
unit-tested with a 1-method fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .signal import SignalEnvelope, SignalIngestResult


class OrchidSignalEmitter(ABC):
    """Single-method interface — emit one envelope, get an ingest
    result back."""

    @abstractmethod
    async def emit(self, envelope: SignalEnvelope) -> SignalIngestResult: ...
