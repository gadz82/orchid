"""Recovery sweep for the publish-then-mark relay queue (§10.3).

When :class:`RelayingSignalQueue.enqueue` flips a signal to
``relay_status='pending_publish'`` and then crashes (or the
publisher itself fails to ack), the row stays in that state forever
unless something re-publishes it.  This producer is the something:

- On :meth:`start` it kicks a periodic background task.
- Each tick reads up to ``batch_size`` signals with
  ``relay_status='pending_publish'`` from the store.
- Each one is re-published via the same :class:`BusPublisher` the
  online queue uses; on success the relay status is flipped to
  ``'published'``.
- Failures stay at ``'pending_publish'`` for the next tick.

The producer takes a :class:`OrchidSignalStore` directly because it
does NOT go through the dispatcher — it's already past the ingest
boundary.  The dispatcher injection in :meth:`start` is captured so
sub-classes that want to re-emit (rare) have it, but the default
implementation does not call it.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from typing import Any

from ...core.events.dispatcher import OrchidSignalDispatcher
from ...core.events.producer import OrchidSignalProducer
from ...core.events.store import OrchidSignalStore
from ..queues.relay import BusPublisher

_logger = logging.getLogger(__name__)


class RelayRecoveryProducer(OrchidSignalProducer):
    """Periodically scan ``signals`` for pending-publish rows and
    re-publish them via the supplied :class:`BusPublisher`.

    The producer is **idempotent under crash**: a partial sweep that
    re-published some rows but not others just continues from where
    the next tick picks up — the bus consumer's at-least-once
    delivery contract handles the duplicates.

    Construction:

    - ``store`` — the same :class:`OrchidSignalStore` the dispatcher
      writes to.  The producer reads pending rows from here and
      flips their status after a successful publish.
    - ``publisher`` — same publisher instance used by the online
      :class:`RelayingSignalQueue`; the recovery sweep is a separate
      caller of the same publish API.
    - ``poll_interval_seconds`` — gap between sweep ticks.  Default
      30s; integrators with tight latency budgets pin it lower.
    - ``batch_size`` — max rows fetched per tick.  Default 100.
    """

    def __init__(
        self,
        *,
        store: OrchidSignalStore,
        publisher: BusPublisher,
        poll_interval_seconds: float = 30.0,
        batch_size: int = 100,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._store = store
        self._publisher = publisher
        self._poll = poll_interval_seconds
        self._batch = batch_size
        self._stopping = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._dispatcher: OrchidSignalDispatcher | None = None

    @property
    def name(self) -> str:
        return "RelayRecoveryProducer"

    # ── Lifecycle ────────────────────────────────────────

    async def start(self, dispatcher: OrchidSignalDispatcher) -> None:
        if self._task is not None:
            raise RuntimeError("RelayRecoveryProducer already started")
        self._dispatcher = dispatcher
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="orchid-relay-recovery")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._poll * 2)
            except TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None

    # ── Sweep body ────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.sweep_once()
            except Exception:
                _logger.exception("RelayRecoveryProducer sweep raised — continuing")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._poll)
                # ``wait`` returned because ``stopping`` was set; exit.
                return
            except TimeoutError:
                # Normal path: the timeout expired, run another sweep.
                continue

    async def sweep_once(self) -> int:
        """Run a single sweep tick.  Returns the number of rows
        successfully republished — useful for tests and for ops
        metrics."""
        pending = await self._fetch_pending()
        if not pending:
            return 0

        republished = 0
        for signal in pending:
            try:
                await self._publisher.publish(signal.signal_id)
            except Exception:
                # Stays at ``pending_publish`` for the next tick.
                _logger.warning(
                    "RelayRecoveryProducer: publish failed for %s — leaving pending_publish for retry",
                    signal.signal_id,
                )
                continue
            try:
                await self._store.update_relay_status(signal.signal_id, status="published")
            except Exception:
                _logger.exception(
                    "RelayRecoveryProducer: update_relay_status failed for %s — at-least-once delivery applies",
                    signal.signal_id,
                )
                continue
            republished += 1
        return republished

    async def _fetch_pending(self) -> list[Any]:
        """Pull up to ``batch_size`` rows with
        ``relay_status='pending_publish'`` ordered oldest-first.

        The :class:`OrchidSignalStore` ABC's ``list`` method does NOT
        carry a relay-status filter (it's an in-memory contract for
        ABC simplicity).  Concrete stores that want to be relayed
        SHOULD expose a ``list_by_relay_status`` accessor; this
        helper duck-types onto it when present and otherwise falls
        back to scanning the recent window — accepting that very
        old un-published rows in stores without the helper might
        slip through until `list(limit=...)` rotates them in.

        Stores that wire the helper land along with the recovery
        producer in production; the in-memory store ships with one
        below.
        """
        # Prefer the explicit accessor when the store provides it.
        accessor = getattr(self._store, "list_by_relay_status", None)
        if accessor is not None:
            try:
                rows = await accessor(status="pending_publish", limit=self._batch)
                return list(rows)
            except Exception:
                _logger.exception("list_by_relay_status raised — falling back to recent-window scan")

        # Fallback: scan the most recent window and filter.
        cutoff = _dt.datetime.now(tz=_dt.UTC) - _dt.timedelta(days=7)
        recent = await self._store.list(since=cutoff, limit=self._batch * 4)
        return [s for s in recent if s.relay_status == "pending_publish"][: self._batch]
