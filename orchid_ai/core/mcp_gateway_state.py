"""
MCP-gateway-state abstractions — persistence for the OAuth
authorization-server role that an MCP gateway (e.g. ``orchid-mcp``)
plays for its inbound clients.

Phase 3 of the auth-centralisation roadmap.  In Phase 1 and 2 the
gateway kept its DCR client registrations, pending auth codes, and
issued access / refresh tokens in memory or in a local JSON file.
That's fine for single-replica deployments but breaks the moment an
operator wants two gateway replicas behind a load balancer: each
replica has its own private copy of the state, so a token minted on
replica A is invisible to replica B.

This module defines three ABCs the gateway can implement against a
shared backend (orchid-api's PostgreSQL / SQLite):

- :class:`OrchidMCPGatewayClientStore` — RFC 7591 DCR client records.
- :class:`OrchidMCPGatewayAuthCodeStore` — pending authorization
  codes with upstream-IdP correlation state.
- :class:`OrchidMCPGatewayTokenStore` — gateway-issued access +
  refresh tokens, with the resolved :class:`OrchidIdentity`-flavoured
  payload that downstream tools consume.

All three share the same lifecycle contract as
:class:`OrchidChatStorage` / :class:`OrchidMCPTokenStore`: construct
with ``__init__(*, dsn: str)``, call :meth:`init_db` on startup,
:meth:`close` on shutdown.  Implementations live in
``orchid_ai.persistence``.

**Direction note.**  The sibling :class:`OrchidMCPTokenStore` /
:class:`OrchidMCPClientRegistrationStore` classes in :mod:`mcp` cover
the **outbound** direction — orchid-api as an MCP *client* calling
external servers that speak MCP 2025-03-26 OAuth.  The ABCs here
cover the **inbound** direction: external MCP clients authenticating
to a gateway that fronts orchid-api.  Keep the distinction in mind
when adding new storage methods.

Uses ONLY stdlib types — safe for ``core/``.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── RFC 7591 DCR client registration (INBOUND) ────────────────────


@dataclass
class OrchidMCPGatewayClient:
    """
    Inbound MCP client that has registered itself with the gateway
    via RFC 7591 Dynamic Client Registration.

    Mirrors the gateway-side TypeScript ``RegisteredClient`` shape
    one-for-one so the HTTP contract is a straight JSON pass-through.
    Fields use OAuth spec names verbatim (``client_id``,
    ``redirect_uris``, …) — readability beats naming consistency here.
    """

    client_id: str
    redirect_uris: list[str]
    grant_types: list[str]
    response_types: list[str]
    token_endpoint_auth_method: str = "none"
    client_name: str = ""
    #: Seconds since epoch.
    created_at: float = field(default_factory=_time.time)


# ── Pending authorization (INBOUND) ───────────────────────────────


@dataclass
class OrchidMCPGatewayAuthCode:
    """
    Authorization code + correlation state carrying an inbound MCP
    client through the ``/authorize`` → upstream IdP → ``/oauth/callback``
    → ``/token`` dance.

    One row per OAuth flow; consumed atomically at ``/token``.
    ``identity`` / ``idp_access_token`` / ``idp_refresh_token`` /
    ``idp_expires_at`` are filled in by the callback handler once
    the upstream exchange completes; they remain empty between
    ``/authorize`` and the callback.
    """

    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    upstream_state: str
    upstream_code_verifier: str
    scopes: list[str]
    #: Seconds since epoch.
    created_at: float = field(default_factory=_time.time)
    #: Client-supplied ``state`` value echoed back on redirect.
    client_state: str = ""
    #: Resolved identity payload (shape owned by the gateway — stored
    #: as an opaque dict so new :class:`OrchidIdentity` fields don't
    #: require a schema migration).
    identity: dict[str, Any] | None = None
    #: Upstream IdP tokens, retained so the gateway can refresh
    #: later without re-prompting the user.
    idp_access_token: str = ""
    idp_refresh_token: str = ""
    idp_expires_at: float = 0.0


# ── Issued gateway token (INBOUND) ────────────────────────────────


@dataclass
class OrchidMCPGatewayToken:
    """
    Access + refresh token the gateway issued to an inbound MCP
    client after a successful OAuth dance.

    ``identity`` carries the gateway's serialized understanding of
    the user (subject, authDomain, …) — stored as an opaque dict so
    the downstream consumers (``OrchidIdentityResolver``) can evolve
    independently of the schema.

    The ``idp_*`` fields carry the **upstream** access / refresh
    tokens the gateway obtained during the browser-based OAuth
    dance.  They live here (in addition to the upstream access
    token that usually doubles as ``identity["bearer"]``) so the
    gateway's ``/token?grant_type=refresh_token`` handler can kick
    off an upstream refresh when the user's bearer is about to
    expire, rather than rotating gateway tokens that still wrap a
    stale upstream credential.  Phase 4 of the auth-centralisation
    roadmap.
    """

    access_token: str
    refresh_token: str
    client_id: str
    subject: str
    identity: dict[str, Any]
    scopes: list[str]
    #: Absolute expiry, seconds since epoch.
    expires_at: float
    #: Upstream IdP access token (typically echoed under
    #: ``identity["bearer"]`` by passthrough-style resolvers, kept
    #: here explicitly so the refresh flow has a single canonical
    #: source).  Empty string when the gateway wasn't given one.
    idp_access_token: str = ""
    #: Upstream IdP refresh token.  When present, a gateway-side
    #: ``/token?grant_type=refresh_token`` call can swap it for a
    #: fresh upstream access/refresh pair before minting new gateway
    #: tokens.  Empty string when the upstream didn't issue one.
    idp_refresh_token: str = ""
    #: Absolute expiry of :attr:`idp_access_token`, seconds since
    #: epoch.  ``0.0`` means "unknown" — the refresh path should
    #: fall back to calling the upstream on every rotation rather
    #: than relying on this field for skew-aware decisions.
    idp_expires_at: float = 0.0


# ── Persistence ABCs ──────────────────────────────────────────────


class OrchidMCPGatewayClientStore(ABC):
    """Persistence contract for :class:`OrchidMCPGatewayClient`.

    Lifecycle follows the house convention: ``__init__(*, dsn: str)``,
    :meth:`init_db` on startup, :meth:`close` on shutdown.
    """

    @abstractmethod
    async def init_db(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def register(self, record: OrchidMCPGatewayClient) -> None:
        """Insert or replace a client registration."""
        ...

    @abstractmethod
    async def get(self, client_id: str) -> OrchidMCPGatewayClient | None:
        """Fetch by ``client_id``.  Returns ``None`` when not found."""
        ...


class OrchidMCPGatewayAuthCodeStore(ABC):
    """Persistence contract for :class:`OrchidMCPGatewayAuthCode`.

    Lifecycle follows the house convention.  Implementations must
    enforce **one-shot** semantics on :meth:`consume` — once a row is
    consumed it must never be returned again, even if another replica
    attempts the same exchange concurrently.  Use atomic
    ``DELETE … RETURNING`` (Postgres) or ``SELECT`` inside a
    transaction followed by ``DELETE`` (SQLite) to achieve this.
    """

    @abstractmethod
    async def init_db(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def put(self, record: OrchidMCPGatewayAuthCode) -> None:
        """Insert a new auth-code record.  Caller guarantees unique ``code``."""
        ...

    @abstractmethod
    async def get_by_upstream_state(
        self,
        upstream_state: str,
    ) -> OrchidMCPGatewayAuthCode | None:
        """Lookup used by ``/oauth/callback`` to correlate with the
        original ``/authorize`` request via the upstream IdP's
        ``state`` echo.
        """
        ...

    @abstractmethod
    async def update(
        self,
        code: str,
        *,
        identity: dict[str, Any] | None = None,
        idp_access_token: str | None = None,
        idp_refresh_token: str | None = None,
        idp_expires_at: float | None = None,
    ) -> None:
        """Patch-style update for the fields ``/oauth/callback``
        fills in post-exchange.  Unspecified arguments leave the
        corresponding column untouched.
        """
        ...

    @abstractmethod
    async def consume(self, code: str) -> OrchidMCPGatewayAuthCode | None:
        """Atomically fetch + delete by ``code``.  Returns the
        record if it existed; subsequent calls with the same ``code``
        return ``None``.
        """
        ...


class OrchidMCPGatewayTokenStore(ABC):
    """Persistence contract for :class:`OrchidMCPGatewayToken`.

    Lifecycle follows the house convention.  Tokens are keyed by
    ``access_token`` (primary) with a secondary unique index on
    ``refresh_token`` for refresh lookups.  Implementations should
    ignore expired rows on lookup — return ``None`` rather than the
    stale record — so downstream consumers don't need TTL bookkeeping.
    """

    @abstractmethod
    async def init_db(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def issue(self, record: OrchidMCPGatewayToken) -> None:
        """Insert a freshly-minted token pair."""
        ...

    @abstractmethod
    async def get_by_access_token(
        self,
        access_token: str,
    ) -> OrchidMCPGatewayToken | None:
        """Lookup used by the gateway on every authenticated MCP
        request.  Must return ``None`` for expired records.
        """
        ...

    @abstractmethod
    async def get_by_refresh_token(
        self,
        refresh_token: str,
    ) -> OrchidMCPGatewayToken | None:
        """Lookup used by the ``refresh_token`` grant flow.
        ``None`` when the refresh token is unknown or its associated
        access token has already expired.
        """
        ...

    @abstractmethod
    async def revoke(self, access_token: str) -> bool:
        """Delete by ``access_token``.  Returns ``True`` when a row
        was removed.
        """
        ...
