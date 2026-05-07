"""Job-side value objects: status, spec, run, retry policy.

A :class:`JobSpec` is the immutable description of what should run; a
:class:`JobRun` is the mutable record of *one attempt* at running it.
Retries become new :class:`JobRun` rows with incremented
``attempt_number`` rather than in-place updates — this preserves the
audit trail and makes the dedup key
``(trigger_id, signal_id, attempt_number)`` unique by construction.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_SCHEDULED = "retry_scheduled"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Per-trigger retry configuration.  Distinct from queue retry —
    queue retry handles process crashes and transient infra failures;
    this policy handles business-level flakiness inside the supervisor
    invocation."""

    max_attempts: int = 0
    backoff: str = "exponential"  # "fixed" | "linear" | "exponential"
    jitter: bool = True
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0

    def delay_for(self, attempt: int) -> float:
        """Compute the wait before the *next* attempt.  ``attempt`` is
        the 1-based attempt number that just failed."""
        if attempt < 1:
            attempt = 1
        if self.backoff == "fixed":
            base = self.initial_delay_seconds
        elif self.backoff == "linear":
            base = self.initial_delay_seconds * attempt
        else:  # exponential, default
            base = self.initial_delay_seconds * (2 ** (attempt - 1))
        capped = min(base, self.max_delay_seconds)
        if not self.jitter:
            return capped
        # Deterministic-ish jitter: 50%–100% of capped.  Real
        # randomness lives in the concrete processor; the policy itself
        # stays pure so unit tests are easy.
        return capped * 0.75


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Immutable description of a job to run."""

    trigger_id: str
    signal_id: _uuid.UUID
    agent_name: str
    prompt: str
    identity_claim: dict[str, Any]
    correlation_id: str | None
    parallelism_key: str  # e.g. "tenant:user" or "tenant:service-account"
    # Run-visibility (§26).  Set by the trigger registry from the
    # trigger config (or its identity-flavour default) at the time
    # the JobSpec is built; immutable for the lifetime of every
    # JobRun spawned from this spec.  ``visibility_user_id`` is the
    # user-of-record for ``actor`` / ``addressed`` visibility and
    # ``None`` for ``tenant`` / ``admin``.
    visibility: str = "admin"
    visibility_user_id: str | None = None
    # §25 chat binding — opt-in metadata read from the matched
    # signal's ``chat_binding`` AND the trigger's
    # ``respect_chat_binding=true`` flag.  The runner re-validates at
    # runtime via ``_resolve_chat_binding`` so a signal that hand-
    # crafts the binding cannot smuggle messages into chats the
    # resolved auth doesn't own.
    chat_binding: dict[str, Any] | None = None


@dataclass(slots=True)
class JobRun:
    """Mutable record of one attempt at running a :class:`JobSpec`."""

    run_id: _uuid.UUID
    spec: JobSpec
    attempt_number: int
    status: JobStatus
    queued_at: _dt.datetime
    started_at: _dt.datetime | None = None
    finished_at: _dt.datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    next_retry_at: _dt.datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
