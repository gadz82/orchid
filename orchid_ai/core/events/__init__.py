"""Pure abstractions for the event-driven activation layer.

This subpackage holds the framework's signal / trigger / job vocabulary
as ABCs and immutable dataclasses with **zero external dependencies** —
the same rule that governs the rest of ``orchid_ai/core/``.  Concrete
implementations (Postgres queue, APScheduler producer, asyncio worker
pool, …) live under ``orchid_ai/events/`` and depend on these ABCs.

The single import direction is::

    orchid_ai/events/  →  orchid_ai/core/events/   (allowed)
    orchid_ai/core/events/  →  orchid_ai/events/   (FORBIDDEN — verified by a test)

A signal flows through the system in two halves:

1. **Ingest path** — synchronous, fast, non-blocking.  A producer
   normalises an external event into a :class:`SignalEnvelope` and calls
   :meth:`OrchidSignalDispatcher.ingest`, which persists the
   :class:`Signal` and enqueues it.  Ingest never matches triggers,
   resolves identity, or invokes the supervisor.
2. **Process path** — asynchronous.  A processor drains the queue,
   resolves the signal's identity claim into an ``OrchidAuthContext``,
   matches triggers, and hands a :class:`JobSpec` to a
   :class:`OrchidJobRunner` which calls the existing supervisor graph.

Identity resolution lives in the processor, not the dispatcher — this
keeps ingest under the latency target and lets ``IdP`` round-trips be
retried on the queue's redelivery contract instead of leaking into the
caller's request budget.
"""

from __future__ import annotations

from .dispatcher import OrchidSignalDispatcher
from .emitter import OrchidSignalEmitter
from .errors import (
    ChatBindingError,
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
    JobRunnerError,
    MintingProbeUnsupportedError,
    OrchidEventsError,
    OrchidIdentityNotMintableError,
    OrchidServiceAccountUnknownError,
    SignalAuthValidationError,
    SignalDuplicateError,
    SignalSourceTypeNotAllowedError,
    SignalSourceUnknownError,
    TriggerMatchError,
    TriggerRegistrationError,
)
from .job import JobRun, JobSpec, JobStatus, RetryPolicy
from .middleware import SignalIngestMiddleware
from .processor import OrchidSignalProcessor
from .producer import OrchidSignalProducer
from .queue import DBTransaction, OrchidSignalQueue, QueuedSignal
from .runner import OrchidJobRunner
from .signal import Signal, SignalEnvelope, SignalIngestResult
from .store import (
    OrchidJobStore,
    OrchidScheduleRecord,
    OrchidScheduleStore,
    OrchidSignalStore,
    OrchidTriggerRecord,
    OrchidTriggerStore,
)
from .trigger import OrchidTrigger, TriggerRegistry

__all__ = [
    "ChatBindingError",
    "ChatBindingForbiddenError",
    "ChatBindingTargetNotFoundError",
    "DBTransaction",
    "JobRun",
    "JobRunnerError",
    "JobSpec",
    "JobStatus",
    "MintingProbeUnsupportedError",
    "OrchidEventsError",
    "OrchidIdentityNotMintableError",
    "OrchidJobRunner",
    "OrchidJobStore",
    "OrchidScheduleRecord",
    "OrchidScheduleStore",
    "OrchidServiceAccountUnknownError",
    "OrchidSignalDispatcher",
    "OrchidSignalEmitter",
    "OrchidSignalProcessor",
    "OrchidSignalProducer",
    "OrchidSignalQueue",
    "OrchidSignalStore",
    "OrchidTrigger",
    "OrchidTriggerRecord",
    "OrchidTriggerStore",
    "QueuedSignal",
    "RetryPolicy",
    "Signal",
    "SignalAuthValidationError",
    "SignalDuplicateError",
    "SignalEnvelope",
    "SignalIngestMiddleware",
    "SignalIngestResult",
    "SignalSourceTypeNotAllowedError",
    "SignalSourceUnknownError",
    "TriggerMatchError",
    "TriggerRegistrationError",
    "TriggerRegistry",
]
