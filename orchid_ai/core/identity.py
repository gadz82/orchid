"""
Abstract identity resolution — consumers provide concrete implementations.

The library depends ONLY on this ABC.  Concrete resolvers (e.g. OAuth,
OIDC, SAML) live in consumer projects and are loaded at runtime via
``settings.identity_resolver_class``.

This module uses ONLY stdlib types — safe for ``core/``.

Two additive methods support the events layer (Pollen + Bloom):

- :meth:`resolve_service_account` returns an ``OrchidAuthContext`` for
  a named service identity (e.g. ``digest-bot``).  Default raises so
  resolvers that don't run service-account triggers don't have to
  implement it.
- :meth:`mint_for_user` mints an ``OrchidAuthContext`` for an
  ``act_as_user`` trigger.  Default raises
  :class:`MintingProbeUnsupportedError`, which the trigger registry
  treats as fatal at boot for any ``act_as_user`` trigger pointed at
  this resolver — surfacing the misconfiguration deterministically
  rather than at first-fire-time.

Both default implementations live here in ``core/`` because the events
ABCs already do.  Concrete consumer resolvers override them as needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .events.errors import (
    MintingProbeUnsupportedError,
    OrchidServiceAccountUnknownError,
)
from .state import OrchidAuthContext


class OrchidIdentityResolver(ABC):
    """
    Resolves a bearer token into an ``OrchidAuthContext``.

    Implementations typically call a platform-specific endpoint
    (e.g. a platform-specific token endpoint, or a standard
    OIDC userinfo endpoint) to validate the token and extract
    identity claims.

    Security contract
    -----------------
    The returned ``OrchidAuthContext.tenant_key`` and ``user_id`` MUST be
    derived **exclusively from data that the upstream IdP attests** —
    typically a verified JWT claim (``iss``, ``sub``, a custom tenant
    claim) or a server-side lookup keyed by the token. Implementations
    MUST NOT trust the ``domain`` argument, request headers, query
    string, or any other client-supplied value when populating these
    fields: those are attacker-controlled, and a mistake here cross-pollutes
    RAG namespaces, chat ownership, and MCP tokens between tenants. The
    ``domain`` parameter is a routing hint for multi-IdP deployments
    only; it never decides ``tenant_key``.

    Role claims (events visibility — see §26)
    -----------------------------------------
    Implementations populate ``OrchidAuthContext.roles`` from their IdP
    claims so the events run-visibility filter can recognise admins.
    The framework reserves ``"admin"``; consumers may add their own
    role names (the framework ignores them, but consumer code may
    use them).  A resolver that does NOT populate ``roles`` produces
    no admins — the safe default.
    """

    @abstractmethod
    async def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
        """
        Validate the bearer token and return a populated OrchidAuthContext.

        Parameters
        ----------
        domain : str
            Tenant domain (e.g. ``acme.example.com``), without protocol.
            Used only to pick which IdP to call — never to populate
            ``tenant_key`` directly. See the class-level security
            contract.
        bearer_token : str
            The raw access token (without ``Bearer `` prefix).

        Raises
        ------
        OrchidIdentityError
            If the token is invalid, expired, or the platform is unreachable.
        """
        ...

    # ── Events-layer extensions (additive, default-raising) ─────

    async def resolve_service_account(self, name: str) -> OrchidAuthContext:
        """Return an ``OrchidAuthContext`` for a named service identity.

        Default raises :class:`OrchidServiceAccountUnknownError`.
        Consumers whose deployment runs ``service_account`` triggers
        override this method to look the name up in their own
        credentials store / vault / IdP.

        The returned context's ``user_id`` should typically be empty
        (service accounts have no user-of-record); the framework's
        run-visibility filter (§26) treats empty ``user_id`` as a
        non-actor for ``actor`` / ``addressed`` visibility.
        """
        raise OrchidServiceAccountUnknownError(name)

    async def mint_for_user(self, tenant_key: str, user_id: str) -> OrchidAuthContext:
        """Mint a fresh ``OrchidAuthContext`` for an ``act_as_user`` Bloom.

        Default raises :class:`MintingProbeUnsupportedError` — the
        trigger registry probes this method at boot for every
        ``act_as_user`` trigger and refuses to start when the resolver
        cannot mint at all.  Consumers that DO support minting
        override this to:

        1. Read the user's stored refresh-token from
           :class:`~orchid_ai.core.mcp.OrchidMCPTokenStore` (or their
           own credentials store).
        2. Refresh against the IdP token endpoint.
        3. Build an ``OrchidAuthContext`` with the resolved
           ``tenant_key`` / ``user_id`` / ``roles`` and the freshly-
           minted ``access_token`` / ``expires_at``.

        Resolvers that have user credentials but cannot mint for a
        SPECIFIC user (e.g. that user has no stored refresh token)
        should raise :class:`OrchidIdentityNotMintableError` directly
        — the registry's mint probe distinguishes "I support minting,
        just not for this sentinel user" (acceptable at boot) from
        "I don't support minting at all" (fatal at boot).
        """
        raise MintingProbeUnsupportedError(type(self).__name__)


class OrchidIdentityError(Exception):
    """Raised when identity resolution fails."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code
