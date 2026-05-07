"""Scheduler producer — turns time into ``cron`` signals.

On :meth:`start`:

1. Load every enabled schedule from :class:`OrchidScheduleStore`.
2. For each, register an APScheduler job whose body calls
   :meth:`OrchidSignalDispatcher.ingest` with a synthetic
   :class:`SignalEnvelope`::

       SignalEnvelope(
           type="cron",
           payload={"schedule_id": <id>, "fire_time": <iso>},
           source=f"scheduler:{schedule_id}",
           occurred_at=now,
           tenant_key="<from identity claim>",
           identity_claim=schedule.identity_claim,
           dedupe_key=f"{schedule_id}:{fire_iso}",
       )

3. After each fire, update :class:`OrchidScheduleStore.record_fire`
   so operators see fresh ``last_fire_at`` / ``next_fire_at`` values.

On :meth:`stop`:

- Shut down the scheduler, waiting on in-flight fires.

The producer does NOT match triggers, resolve identity, or invoke
the supervisor — those belong to the processor on the dequeue side.
The dispatcher's outbox commits a real ``signals`` row plus a real
``signal_queue`` row, so a process restart between fire-time and
processor-pickup is safe.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Callable

from ...core.events.dispatcher import OrchidSignalDispatcher
from ...core.events.errors import SignalDuplicateError
from ...core.events.producer import OrchidSignalProducer
from ...core.events.signal import SignalEnvelope
from ...core.events.store import OrchidScheduleStore
from ..schedulers.apscheduler import APSchedulerBackend

_logger = logging.getLogger(__name__)


class SchedulerProducer(OrchidSignalProducer):
    """Cron / interval signal source backed by APScheduler."""

    def __init__(
        self,
        *,
        schedule_store: OrchidScheduleStore,
        backend: APSchedulerBackend | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
        timezone: str = "UTC",
    ) -> None:
        self._schedule_store = schedule_store
        self._backend = backend or APSchedulerBackend(timezone=timezone)
        self._clock = clock or _default_clock
        self._dispatcher: OrchidSignalDispatcher | None = None
        self._registered: set[str] = set()

    @property
    def name(self) -> str:
        return "SchedulerProducer"

    @property
    def backend(self) -> APSchedulerBackend:
        """Test/integration accessor for the underlying APScheduler
        wrapper.  Not part of the producer ABC."""
        return self._backend

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self, dispatcher: OrchidSignalDispatcher) -> None:
        self._dispatcher = dispatcher
        await self._backend.start()
        await self.refresh()

    async def stop(self) -> None:
        await self._backend.shutdown(wait=True)
        self._registered.clear()

    async def refresh(self) -> None:
        """Re-sync registered jobs against the current
        ``schedule_store`` contents.  Called once at start; can be
        called again any time a schedule is added / removed / toggled
        without restarting the producer."""
        if self._dispatcher is None:
            raise RuntimeError("SchedulerProducer.refresh called before start(dispatcher)")

        records = list(await self._schedule_store.list())
        live_ids: set[str] = set()
        for record in records:
            if not record.enabled:
                continue
            if record.cron is None and record.interval_seconds is None:
                _logger.warning(
                    "schedule %s has neither cron nor interval — skipping",
                    record.schedule_id,
                )
                continue
            live_ids.add(record.schedule_id)
            callback = self._make_callback(
                schedule_id=record.schedule_id,
                identity_claim=record.identity_claim,
                tenant_key=_tenant_key_from_claim(record.identity_claim),
            )
            if record.cron is not None:
                self._backend.add_cron(
                    schedule_id=record.schedule_id,
                    cron=record.cron,
                    callback=callback,
                )
            else:
                # Interval mode — APScheduler defaults to first fire
                # at "now + interval"; for ergonomics we let tests
                # opt into an immediate first fire by passing a tiny
                # interval (e.g. 0.1s won't work because we use int
                # seconds; tests use 1-second intervals + 50ms
                # next_run_time hints).  Default behaviour: standard.
                self._backend.add_interval(
                    schedule_id=record.schedule_id,
                    seconds=int(record.interval_seconds),
                    callback=callback,
                )
            self._registered.add(record.schedule_id)

        # Cancel jobs that are no longer enabled / present.
        stale = self._registered - live_ids
        for schedule_id in stale:
            self._backend.remove(schedule_id)
            self._registered.discard(schedule_id)

    # ── Job body factory ─────────────────────────────────────

    def _make_callback(
        self,
        *,
        schedule_id: str,
        identity_claim: dict[str, Any],
        tenant_key: str,
    ) -> Callable[[], Any]:
        """Build the per-schedule async fire callback."""

        async def _fire() -> None:
            assert self._dispatcher is not None  # set in start()
            now = self._clock()
            fire_iso = now.isoformat()
            envelope = SignalEnvelope(
                type="cron",
                payload={
                    "schedule_id": schedule_id,
                    "fire_time": fire_iso,
                },
                source=f"scheduler:{schedule_id}",
                occurred_at=now,
                tenant_key=tenant_key,
                # ``cron`` signals don't carry a user_id — identity is
                # in the claim and resolved by the processor.
                identity_claim=dict(identity_claim) if identity_claim else None,
                dedupe_key=f"{schedule_id}:{fire_iso}",
            )
            try:
                await self._dispatcher.ingest(envelope)
            except SignalDuplicateError:
                # Two scheduler instances racing on the same fire
                # is harmless — the dedupe key holds, the second one
                # is a no-op.  Log at debug so production noise
                # stays low.
                _logger.debug(
                    "scheduler %s: duplicate cron fire at %s — already ingested",
                    schedule_id,
                    fire_iso,
                )
                return
            except Exception:
                _logger.exception(
                    "scheduler %s: dispatcher.ingest failed at %s",
                    schedule_id,
                    fire_iso,
                )
                return

            next_fire = self._backend.get_next_fire(schedule_id)
            try:
                await self._schedule_store.record_fire(schedule_id, last_fire_at=now, next_fire_at=next_fire)
            except Exception:
                # Operator-visible bookkeeping; not worth crashing
                # the producer over.
                _logger.exception("scheduler %s: record_fire failed", schedule_id)

        return _fire


# ── Helpers ──────────────────────────────────────────────────


def _default_clock() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _tenant_key_from_claim(identity_claim: dict[str, Any]) -> str:
    """Schedules carry an identity claim (service-account /
    addressed-to-user / act-as-user) but the trigger consumer needs
    a ``tenant_key`` on the envelope.  Pull it out of the claim's
    ``tenant_key`` field if present, otherwise fall back to
    ``"default"`` — same convention as
    :attr:`OrchidAuthContext.tenant_key`.

    Today the claim shape from
    ``orchid_ai.config.schema_events`` doesn't formally include a
    ``tenant_key`` slot, but integrators routinely tag the claim with
    one when they ``upsert`` the schedule programmatically (e.g.
    multi-tenant SaaS deployments).  Honouring the slot here means
    those deployments don't have to override the producer."""
    candidate = identity_claim.get("tenant_key") if identity_claim else None
    if isinstance(candidate, str) and candidate:
        return candidate
    return "default"
