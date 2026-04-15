"""
MCP client abstraction.

Concrete implementations live in agents/src/mcp/:
  - StreamableHttpMCPClient (production)

Domain types for per-server OAuth token management:
  - MCPTokenRecord — stored token for a (server, tenant, user) triple
  - MCPTokenStore — persistence ABC for OAuth tokens
  - MCPAuthRequiredError — raised when OAuth authorization is needed

The agent never knows which transport or auth mode is being used.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .state import AuthContext


# ── Segregated Interfaces (ISP) ────────────────────────────


class MCPToolCaller(ABC):
    """Minimal interface for invoking MCP tools — most consumers only need this."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: AuthContext,
    ) -> MCPToolResult:
        """Invoke a tool on the remote/local MCP server."""
        ...

    @property
    @abstractmethod
    def server_url(self) -> str:
        """Base URL of the MCP server (for logging/debugging)."""
        ...


class MCPDiscoverable(ABC):
    """Interface for discovering MCP server capabilities."""

    @abstractmethod
    async def list_tools(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available tools on the MCP server."""
        ...

    @abstractmethod
    async def list_prompts(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available prompts on the MCP server."""
        ...

    @abstractmethod
    async def list_resources(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available resources on the MCP server."""
        ...

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: AuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template and return its messages."""
        ...

    @abstractmethod
    async def read_resource(self, uri: str, auth: AuthContext) -> str:
        """Read the content of a resource by URI."""
        ...


@dataclass
class MCPToolResult:
    """Normalised result from an MCP tool call."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        """Convenience: concatenate all text content blocks."""
        return "\n".join(item.get("text", "") for item in self.content if item.get("type") == "text")


class MCPClient(MCPToolCaller, MCPDiscoverable, ABC):
    """
    Combined MCP client interface.

    Extends both ``MCPToolCaller`` (tool invocation) and ``MCPDiscoverable``
    (capability discovery).  Code that only needs tool calling should
    depend on ``MCPToolCaller`` instead for better interface segregation.
    """

    pass


# ── Per-server OAuth token management ────────────────────────


class MCPAuthRequiredError(Exception):
    """Raised when an MCP server requires OAuth but the user has not authorized.

    Caught at the fault-isolation boundaries in ``mcp_dispatcher.py``
    (the existing ``except Exception`` blocks) — one unauthorized server
    must not crash the entire agent.
    """

    def __init__(self, server_name: str) -> None:
        super().__init__(f"OAuth authorization required for MCP server '{server_name}'")
        self.server_name = server_name


@dataclass
class MCPTokenRecord:
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


class MCPTokenStore(ABC):
    """Persistence contract for per-server OAuth tokens.

    Follows the same lifecycle and factory patterns as ``ChatStorage``:
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
    ) -> MCPTokenRecord | None:
        """Retrieve a stored token, or ``None`` if not found."""
        ...

    @abstractmethod
    async def save_token(self, record: MCPTokenRecord) -> None:
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
    ) -> list[MCPTokenRecord]:
        """List all tokens for a given tenant + user."""
        ...
