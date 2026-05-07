"""Exception hierarchy for the events subsystem.

Two parent classes exist intentionally:

- :class:`OrchidEventsError` — root for everything raised from the
  Pollen + Bloom pipeline (queue / producer / processor / runner).
- The identity-related errors (:class:`OrchidServiceAccountUnknownError`,
  :class:`OrchidIdentityNotMintableError`) live here too so the events
  pipeline can raise them without importing from
  :mod:`orchid_ai.core.identity` at module-import time, but they remain
  part of the events vocabulary.

The ``OrchidIdentityResolver`` ABC catches *both* defaults via the
events package, so concrete resolvers in consumer projects don't need
to know which subpackage owns the symbol.
"""

from __future__ import annotations


class OrchidEventsError(Exception):
    """Root exception for the events subsystem."""


class SignalDuplicateError(OrchidEventsError):
    """Raised when a producer ingests a signal that violates
    ``UNIQUE (source, dedupe_key)``.  The dispatcher normally swallows
    this and returns the original ``signal_id`` with
    ``deduplicated=True``; concrete stores raise it so the dispatcher
    can detect the collision."""


class SignalSourceUnknownError(OrchidEventsError):
    """Raised when an HTTP ingestion request names a source that is not
    registered in ``signal_sources``."""


class SignalSourceTypeNotAllowedError(OrchidEventsError):
    """Raised when a signal's declared ``type`` is not in the source's
    allowlist.  Surfaces as ``403`` at the API boundary."""


class SignalAuthValidationError(OrchidEventsError):
    """Raised by a ``SignalAuthValidator`` when HMAC / bearer / mTLS
    checks fail.  Surfaces as ``401`` at the API boundary."""


class TriggerRegistrationError(OrchidEventsError):
    """Raised at boot when a trigger's YAML config fails validation
    (unknown agent, unparseable cron, resolver lacks ``mint_for_user``,
    …).  This must surface at startup, never at first-fire time."""


class TriggerMatchError(OrchidEventsError):
    """Raised when a JMESPath ``when:`` expression evaluation fails at
    runtime — typically a malformed payload that didn't match the
    expression's expected shape."""


class JobRunnerError(OrchidEventsError):
    """Raised by an ``OrchidJobRunner`` to signal a failure inside the
    supervisor invocation.  ``retryable=True`` lets the processor's
    retry layer schedule another attempt; ``retryable=False`` is
    terminal."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class OrchidServiceAccountUnknownError(OrchidEventsError):
    """Default raise from ``OrchidIdentityResolver.resolve_service_account``
    when the resolver does not know the requested service-account name.

    Concrete resolvers override the method to look the name up in their
    own credentials store; the raising default keeps the ABC pure."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown service account: {name!r}")
        self.name = name


class OrchidIdentityNotMintableError(OrchidEventsError):
    """Default raise from ``OrchidIdentityResolver.mint_for_user`` when
    the resolver cannot mint a fresh ``OrchidAuthContext`` for the
    given user (no stored refresh token, IdP doesn't support
    token-exchange, etc.).

    Triggers with ``identity.mode == "act_as_user"`` probe ``mint_for_user``
    at registration time and refuse to start when the resolver raises a
    :class:`MintingProbeUnsupportedError` (subclass below) — the regular
    :class:`OrchidIdentityNotMintableError` only signals 'this user has
    no credentials right now', which is fine at boot."""

    def __init__(self, tenant_key: str, user_id: str) -> None:
        super().__init__(f"cannot mint identity for {tenant_key}:{user_id}")
        self.tenant_key = tenant_key
        self.user_id = user_id


class MintingProbeUnsupportedError(OrchidIdentityNotMintableError):
    """Raised by a resolver that cannot mint identities at all (as
    opposed to 'cannot mint for this specific user').  Trigger
    registration treats this as fatal and refuses to start."""

    def __init__(self, resolver_class: str) -> None:
        # Reuse parent constructor with sentinel keys so the message
        # still fits the parent's contract.
        super().__init__(tenant_key="__probe__", user_id="__probe__")
        self.resolver_class = resolver_class
        self.args = (f"resolver {resolver_class!r} does not support mint_for_user",)


# ── Chat-binding errors (§25) ─────────────────────────────────


class ChatBindingError(OrchidEventsError):
    """Root for chat-binding failures.  Always non-retryable — a
    binding that doesn't resolve at attempt N won't resolve at N+1
    either, and we don't want failed bindings looping in the queue."""


class ChatBindingTargetNotFoundError(ChatBindingError):
    """Raised when ``chat_binding.chat_id`` references a chat that no
    longer exists in :class:`OrchidChatStorage`.  The run finishes
    ``status=failed`` with a clear error message; per §25.4 we do
    NOT post the on_failure error message in this case (the runner
    cannot trust a chat target it just rejected)."""

    def __init__(self, chat_id: str) -> None:
        super().__init__(f"chat-binding target chat {chat_id!r} not found")
        self.chat_id = chat_id


class ChatBindingForbiddenError(ChatBindingError):
    """Raised when the resolved auth is not allowed to write to the
    target chat (cross-user smuggling attempt).  Run finishes
    ``status=failed``; same no-post-on-error rule as the not-found
    case (§25.4)."""

    def __init__(self, chat_id: str, user_id: str) -> None:
        super().__init__(f"auth user {user_id!r} cannot write to chat {chat_id!r}")
        self.chat_id = chat_id
        self.user_id = user_id
