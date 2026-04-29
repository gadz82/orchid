"""Per-server dynamic-client-registration metadata (RFC 7591 + RFC 8414).

These types belong to the *server* identity — one row per MCP server,
not per user — so they're segregated from the per-user
:mod:`mcp_tokens` module.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OrchidMCPClientRegistration:
    """Per-server authorization metadata discovered at runtime.

    Populated by the MCP 2025-03-26 discovery chain:

      1. 401 from the MCP server with
         ``WWW-Authenticate: Bearer resource_metadata="…"`` (RFC 9728).
      2. GET that URL → protected-resource metadata → pick an entry
         from ``authorization_servers``.
      3. GET the auth server's ``/.well-known/oauth-authorization-server``
         (RFC 8414) → endpoints + supported grant types / auth methods
         / PKCE methods / scopes.
      4. POST to ``registration_endpoint`` with
         ``redirect_uris`` + metadata (RFC 7591) → receive
         ``client_id`` and optionally ``client_secret``.

    The resulting record is persisted ONCE per MCP server (not per
    user) so subsequent container restarts reuse the same registration
    rather than creating a new one on every boot (which most
    authorization servers rate-limit).

    Per-USER access / refresh tokens remain in
    :class:`OrchidMCPTokenRecord` / :class:`OrchidMCPTokenStore`.
    """

    server_name: str
    #: From RFC 8414 metadata.
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str = ""  # kept so refreshed re-registration is possible
    issuer: str = ""
    scopes_supported: str = ""  # space-separated per OAuth 2.0 convention
    token_endpoint_auth_methods_supported: str = "client_secret_post"
    #: From RFC 7591 registration response.
    client_id: str = ""
    client_secret: str = ""
    client_id_issued_at: float = 0.0
    client_secret_expires_at: float = 0.0  # 0 == non-expiring per RFC 7591
    #: House-keeping.
    created_at: float = field(default_factory=_time.time)
    updated_at: float = field(default_factory=_time.time)

    @property
    def uses_basic_auth(self) -> bool:
        """True when the authorization server advertised Basic auth.

        Drives whether token exchanges send credentials via
        HTTP Basic or via the request body (``client_secret_post``).
        """
        return "client_secret_basic" in self.token_endpoint_auth_methods_supported

    @property
    def is_public_client(self) -> bool:
        """True when DCR returned no secret (PKCE-only, public client)."""
        return not self.client_secret


class OrchidMCPClientRegistrationStore(ABC):
    """Persistence contract for :class:`OrchidMCPClientRegistration`.

    Mirrors :class:`OrchidMCPTokenStore` lifecycle: construct with
    ``__init__(*, dsn: str)``, ``init_db()`` on startup, ``close()``
    on shutdown.  Registrations are keyed by ``server_name`` alone —
    the registered client is a property of the MCP server, not of the
    user, so one row serves every user of a given installation.
    """

    @abstractmethod
    async def init_db(self) -> None:
        """Open connections and run migrations."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release connections and resources."""
        ...

    @abstractmethod
    async def get(self, server_name: str) -> OrchidMCPClientRegistration | None:
        """Retrieve a stored registration, or ``None`` if not found."""
        ...

    @abstractmethod
    async def save(self, record: OrchidMCPClientRegistration) -> None:
        """Insert or update (upsert) a registration row."""
        ...

    @abstractmethod
    async def delete(self, server_name: str) -> bool:
        """Delete a registration.  Returns ``True`` if a row was removed."""
        ...
