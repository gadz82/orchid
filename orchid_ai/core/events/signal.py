"""Immutable signal value objects.

A :class:`SignalEnvelope` is what a producer hands to
:meth:`OrchidSignalDispatcher.ingest` — it lacks a ``signal_id`` and a
``persisted_at`` timestamp because the store assigns those during
ingest.  A :class:`Signal` is the persisted form, returned by store
reads and consumed by the processor.

Both are ``frozen=True`` dataclasses with ``slots=True`` — value
semantics, no accidental mutation, fast attribute access.

The ``identity_claim`` field is a serialised discriminated union (the
``ServiceAccountIdentity`` / ``AddressedToUserIdentity`` /
``ActAsUserIdentity`` Pydantic models from
``orchid_ai/config/schema_events.py``).  ``core/events/`` cannot import
Pydantic, so we carry the claim as a plain ``dict`` — the processor
reconstructs the typed model when it needs to dispatch on ``mode``.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalEnvelope:
    """The shape an emitter hands to the dispatcher.

    ``occurred_at`` is the *event* timestamp (when the underlying thing
    happened in the source system); ``persisted_at`` (only on
    :class:`Signal`) is when the framework committed it.  These can
    differ by minutes for retried producers or by days for replayed
    history.

    ``chat_binding`` is the opt-in opt-in §25 binding to a real user
    chat — when present and the matched trigger has
    ``respect_chat_binding=true`` and the resolved auth has write
    permission on the target chat, the resulting Bloom's final
    ``AIMessage`` lands in that chat with ``metadata.origin="bloom"``.
    Without all three conditions the binding is ignored.  Carried as
    a plain dict because ``core/events/`` cannot import Pydantic; the
    structured model lives in
    :mod:`orchid_ai.config.schema_events.ChatBinding`.
    """

    type: str
    payload: dict[str, Any]
    source: str
    occurred_at: _dt.datetime
    tenant_key: str
    user_id: str | None = None
    correlation_id: str | None = None
    dedupe_key: str | None = None
    identity_claim: dict[str, Any] | None = None
    chat_binding: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Signal:
    """A persisted signal.  Returned by ``OrchidSignalStore.get``.

    Field order mirrors :class:`SignalEnvelope` so a store can build one
    by spreading an envelope plus the two ingest-assigned fields.
    Defined as a sibling rather than a subclass because frozen
    dataclass inheritance interacts awkwardly with default values when
    new required fields are added.
    """

    type: str
    payload: dict[str, Any]
    source: str
    occurred_at: _dt.datetime
    tenant_key: str
    signal_id: _uuid.UUID
    persisted_at: _dt.datetime
    user_id: str | None = None
    correlation_id: str | None = None
    dedupe_key: str | None = None
    identity_claim: dict[str, Any] | None = None
    chat_binding: dict[str, Any] | None = None
    relay_status: str = "committed"

    @classmethod
    def from_envelope(
        cls,
        envelope: SignalEnvelope,
        *,
        signal_id: _uuid.UUID,
        persisted_at: _dt.datetime,
        relay_status: str = "committed",
    ) -> Signal:
        return cls(
            type=envelope.type,
            payload=envelope.payload,
            source=envelope.source,
            occurred_at=envelope.occurred_at,
            tenant_key=envelope.tenant_key,
            signal_id=signal_id,
            persisted_at=persisted_at,
            user_id=envelope.user_id,
            correlation_id=envelope.correlation_id,
            dedupe_key=envelope.dedupe_key,
            identity_claim=dict(envelope.identity_claim) if envelope.identity_claim else None,
            chat_binding=dict(envelope.chat_binding) if envelope.chat_binding else None,
            relay_status=relay_status,
        )

    def as_match_dict(self) -> dict[str, Any]:
        """Expose the signal as a flat dict for JMESPath ``when:``
        evaluation.  Triggers see ``signal.type``, ``signal.user_id``,
        ``payload.foo``, ``source``, ``tenant_key`` — all from this
        single mapping.

        ``chat_binding`` is intentionally **not** exposed here — letting
        triggers filter on it would let an attacker steer their signal
        to a trigger that opts in to chat writes by hand-crafting the
        binding, which the §25 design explicitly prohibits.  The
        binding is consumed at runtime by ``GraphJobRunner``, never by
        the trigger match phase.
        """
        return {
            "signal": {
                "id": str(self.signal_id),
                "type": self.type,
                "source": self.source,
                "tenant_key": self.tenant_key,
                "user_id": self.user_id,
                "correlation_id": self.correlation_id,
                "occurred_at": self.occurred_at.isoformat(),
            },
            "payload": dict(self.payload),
            "type": self.type,
            "source": self.source,
            "tenant_key": self.tenant_key,
        }


@dataclass(frozen=True, slots=True)
class SignalIngestResult:
    """Returned by :meth:`OrchidSignalDispatcher.ingest`.

    ``deduplicated`` is ``True`` when the store already had a row with
    the same ``(source, dedupe_key)`` — the existing ``signal_id`` is
    returned unchanged and ``queue_msg_id`` is ``None``."""

    signal_id: _uuid.UUID
    queue_msg_id: str | None
    deduplicated: bool = field(default=False)
