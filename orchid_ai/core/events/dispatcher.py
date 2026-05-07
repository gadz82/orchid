"""Ingest funnel — turns a :class:`SignalEnvelope` into a persisted
:class:`Signal` plus a queue message.

The dispatcher is small on purpose:

1. Run middleware in order (each can mutate-and-return the envelope or
   raise to short-circuit).
2. Open a queue transaction (or a no-op when the queue is non-tx).
3. Insert the signal row.  On ``UNIQUE (source, dedupe_key)`` collision
   return the existing ``signal_id`` with ``deduplicated=True`` and
   skip the enqueue.
4. Enqueue under the same transaction handle.
5. Commit, return a :class:`SignalIngestResult`.

It does **not** match triggers, resolve identity, or invoke the
supervisor — those live in the processor.  Keeping the dispatcher
narrow is what lets ``ingest`` complete in well under the latency
target (a single ``INSERT … ON CONFLICT`` plus an enqueue, in one
transaction).
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid as _uuid
from typing import Iterable

from .errors import SignalDuplicateError
from .middleware import SignalIngestMiddleware
from .queue import OrchidSignalQueue
from .signal import Signal, SignalEnvelope, SignalIngestResult
from .store import OrchidSignalStore

_logger = logging.getLogger(__name__)


class OrchidSignalDispatcher:
    """Synchronous, fast, non-blocking-with-respect-to-processing
    ingest funnel."""

    def __init__(
        self,
        *,
        store: OrchidSignalStore,
        queue: OrchidSignalQueue,
        middleware: Iterable[SignalIngestMiddleware] | None = None,
        clock: "object" = None,
    ) -> None:
        self._store = store
        self._queue = queue
        self._middleware: list[SignalIngestMiddleware] = list(middleware or [])
        # ``clock`` is an injectable callable returning a tz-aware
        # datetime, so unit tests can pin ``persisted_at``.  Defaults
        # to ``datetime.datetime.now(tz=UTC)`` — kept as a callable
        # rather than calling at import time to avoid a frozen module-
        # load timestamp.
        self._clock = clock or self._default_clock

    @staticmethod
    def _default_clock() -> _dt.datetime:
        return _dt.datetime.now(tz=_dt.UTC)

    async def ingest(self, envelope: SignalEnvelope) -> SignalIngestResult:
        """The single entry point.  See module docstring for the
        five-step sequence.  Always returns — the only way to abort
        ingestion is for a middleware to raise."""
        for mw in self._middleware:
            envelope = await mw(envelope)

        signal_id = _uuid.uuid4()
        persisted_at = self._clock()
        signal = Signal.from_envelope(envelope, signal_id=signal_id, persisted_at=persisted_at)

        async with self._queue.transaction() as tx:
            try:
                stored = await self._store.insert(signal, tx=tx)
            except SignalDuplicateError as exc:
                _logger.debug(
                    "ingest dedup hit source=%s dedupe_key=%s existing=%s",
                    envelope.source,
                    envelope.dedupe_key,
                    exc.args[0] if exc.args else "?",
                )
                existing_id = await self._store.find_by_dedupe(
                    source=envelope.source,
                    dedupe_key=envelope.dedupe_key,
                )
                if existing_id is None:
                    raise
                return SignalIngestResult(signal_id=existing_id, queue_msg_id=None, deduplicated=True)
            queue_msg_id = await self._queue.enqueue(stored.signal_id, tx=tx)

        return SignalIngestResult(signal_id=stored.signal_id, queue_msg_id=queue_msg_id, deduplicated=False)
