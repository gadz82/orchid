"""Pre-persistence ingest middleware — the ABC.

A middleware is an async callable that receives a
:class:`SignalEnvelope` and returns either:

- a (possibly mutated) :class:`SignalEnvelope`, OR
- raises an exception to short-circuit ingestion.

Middlewares run *before* the signal is persisted, so they can:

- redact or transform the payload (PII scrubbing, field masking),
- decorate the envelope with tracing / correlation IDs,
- reject the envelope (rate-limit, schema gate) by raising.

They cannot drop the envelope silently — every ingest call has to
either return a result or raise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .signal import SignalEnvelope


class SignalIngestMiddleware(ABC):
    """Async callable invoked in order by the dispatcher."""

    @abstractmethod
    async def __call__(self, envelope: SignalEnvelope) -> SignalEnvelope: ...
