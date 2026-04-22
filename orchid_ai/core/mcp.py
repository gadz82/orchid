"""
MCP client abstraction.

Concrete implementations live in ``mcp/``:
  - StreamableHttpMCPClient (production)

Domain types for per-server OAuth (MCP Authorization 2025-03-26):
  - OrchidMCPTokenRecord — per-user stored access+refresh token
  - OrchidMCPTokenStore — persistence ABC for user tokens
  - OrchidMCPClientRegistration — per-server discovered metadata +
    dynamically-registered client credentials (RFC 7591 DCR)
  - OrchidMCPClientRegistrationStore — persistence ABC for the above
  - OrchidMCPAuthRequiredError — raised when OAuth authorization is needed

The agent never knows which transport or auth mode is being used.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .state import OrchidAuthContext


# ── Segregated Interfaces (ISP) ────────────────────────────


class OrchidMCPToolCaller(ABC):
    """Minimal interface for invoking MCP tools — most consumers only need this."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: OrchidAuthContext,
    ) -> OrchidMCPToolResult:
        """Invoke a tool on the remote/local MCP server."""
        ...

    @property
    @abstractmethod
    def server_url(self) -> str:
        """Base URL of the MCP server (for logging/debugging)."""
        ...


class OrchidMCPDiscoverable(ABC):
    """Interface for discovering MCP server capabilities."""

    @abstractmethod
    async def list_tools(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available tools on the MCP server."""
        ...

    @abstractmethod
    async def list_prompts(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available prompts on the MCP server."""
        ...

    @abstractmethod
    async def list_resources(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available resources on the MCP server."""
        ...

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: OrchidAuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template and return its messages."""
        ...

    @abstractmethod
    async def read_resource(self, uri: str, auth: OrchidAuthContext) -> str:
        """Read the content of a resource by URI."""
        ...


@dataclass
class OrchidMCPToolResult:
    """Normalised result from an MCP tool call."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        """Convenience: concatenate all text content blocks."""
        return "\n".join(item.get("text", "") for item in self.content if item.get("type") == "text")


class OrchidMCPClient(OrchidMCPToolCaller, OrchidMCPDiscoverable, ABC):
    """
    Combined MCP client interface.

    Extends both ``OrchidMCPToolCaller`` (tool invocation) and ``OrchidMCPDiscoverable``
    (capability discovery).  Code that only needs tool calling should
    depend on ``OrchidMCPToolCaller`` instead for better interface segregation.
    """

    pass


# ── Per-server OAuth token management ────────────────────────


class OrchidMCPAuthRequiredError(Exception):
    """Raised when an MCP server requires OAuth but the user has not authorized.

    Caught at the fault-isolation boundaries in ``mcp_dispatcher.py``
    (the existing ``except Exception`` blocks) — one unauthorized server
    must not crash the entire agent.
    """

    def __init__(self, server_name: str) -> None:
        super().__init__(f"OAuth authorization required for MCP server '{server_name}'")
        self.server_name = server_name


class OrchidMCPDiscoveryError(Exception):
    """Raised when MCP authorization discovery (RFC 9728 / RFC 8414 / RFC 7591) fails.

    Signals a spec-compliance issue with the MCP server or its
    authorization server: missing ``WWW-Authenticate`` header on 401,
    malformed protected-resource metadata, auth-server metadata without
    a ``registration_endpoint``, DCR rejection, etc.  Distinct from
    :class:`OrchidMCPAuthRequiredError` which signals a completely
    normal "user hasn't authenticated yet" state.
    """

    def __init__(self, server_name: str, reason: str) -> None:
        super().__init__(f"MCP authorization discovery failed for '{server_name}': {reason}")
        self.server_name = server_name
        self.reason = reason


# ── Dynamic client registration (RFC 7591 + RFC 8414) ─────────


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


@dataclass
class OrchidMCPTokenRecord:
    """Stored OAuth token for a ``(server_name, tenant_id, user_id)`` triple.

    Mirrors the CLI's ``StoredToken`` pattern but adds multi-tenancy keys
    so tokens are isolated per user and per tenant.
    """

    server_name: str
    tenant_id: str
    user_id: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0  # epoch seconds; 0 = no expiry info
    scopes: str = ""
    created_at: float = field(default_factory=_time.time)
    updated_at: float = field(default_factory=_time.time)

    @property
    def is_expired(self) -> bool:
        """True when the access token has expired."""
        return self.expires_at > 0 and _time.time() >= self.expires_at

    @property
    def is_refresh_available(self) -> bool:
        """True when a refresh token is present."""
        return bool(self.refresh_token)

    @property
    def bearer_header(self) -> dict[str, str]:
        """Ready-to-use Authorization header."""
        return {"Authorization": f"Bearer {self.access_token}"}


class OrchidMCPTokenStore(ABC):
    """Persistence contract for per-server OAuth tokens.

    Follows the same lifecycle and factory patterns as ``OrchidChatStorage``:
    construct with ``__init__(*, dsn: str)``, call ``init_db()`` on
    startup, ``close()`` on shutdown.

    Tokens are keyed by ``(tenant_id, user_id, server_name)``.
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
    async def get_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> OrchidMCPTokenRecord | None:
        """Retrieve a stored token, or ``None`` if not found."""
        ...

    @abstractmethod
    async def save_token(self, record: OrchidMCPTokenRecord) -> None:
        """Insert or update (upsert) a token record."""
        ...

    @abstractmethod
    async def delete_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> bool:
        """Delete a stored token.  Returns ``True`` if a row was removed."""
        ...

    @abstractmethod
    async def list_tokens(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidMCPTokenRecord]:
        """List all tokens for a given tenant + user."""
        ...
