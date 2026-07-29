"""Triggers map signals to job specs.

A trigger is a *pure function* of :class:`Signal` →
:class:`JobSpec` | ``None`` plus a small bag of metadata (id, retry
policy, parallelism mode, identity claim).  Pure means **no I/O** in
``matches`` or ``build_job_spec`` — the only interesting work is the
JMESPath ``when:`` evaluation, which is deterministic.

The :class:`TriggerRegistry` is the surface the processor talks to: it
asks 'which triggers fire for this signal?' and gets back an iterable.
The reference implementation in ``orchid_ai/events/registry.py`` is an
in-memory list scan; integrators with very large trigger sets can
substitute a backend that indexes by ``signal.type``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .job import JobSpec, RetryPolicy
from .signal import Signal


class OrchidTrigger(ABC):
    """A single trigger.  Lifecycle: instantiated at boot from YAML,
    discarded only on config reload."""

    @property
    @abstractmethod
    def trigger_id(self) -> str: ...

    @property
    @abstractmethod
    def parallelism(self) -> str:
        """One of ``per_user`` (default), ``per_tenant``, or
        ``unbounded``.  Drives the processor's serialisation key."""

    @property
    @abstractmethod
    def retry_policy(self) -> RetryPolicy: ...

    @property
    @abstractmethod
    def identity_claim(self) -> dict[str, object]:
        """The serialised identity claim (the same shape stored on
        ``Signal.identity_claim`` when the trigger fires synthetically
        for a schedule).  ``mode`` is one of ``service_account``,
        ``addressed_to_user``, ``act_as_user``."""

    @abstractmethod
    def matches(self, signal: Signal) -> bool:
        """Return ``True`` iff this trigger should fire for the signal.
        Must be deterministic and side-effect-free."""

    @abstractmethod
    def build_job_spec(self, signal: Signal) -> JobSpec:
        """Build the job spec.  Called only after :meth:`matches`
        returned ``True``.  Pure function of the signal."""


class TriggerRegistry(ABC):
    """Read/write surface for the live trigger set."""

    @abstractmethod
    def register(self, trigger: OrchidTrigger) -> None:
        """Register a trigger.  Implementations MUST run any
        registration-time validation (cron parseability, agent
        existence, ``mint_for_user`` probe for ``act_as_user``) and
        raise ``TriggerRegistrationError`` on failure."""

    @abstractmethod
    def find_matches(self, signal: Signal) -> Iterable[OrchidTrigger]:
        """Yield every trigger whose :meth:`OrchidTrigger.matches`
        returns ``True`` for the signal."""

    @abstractmethod
    def get(self, trigger_id: str) -> OrchidTrigger | None: ...

    @abstractmethod
    def all(self) -> Iterable[OrchidTrigger]: ...
