"""Per-user MCP OAuth token persistence contract.

Distinct from :mod:`mcp_registration` (which is keyed per server) — a
token record is owned by the ``(tenant_id, user_id, server_name)``
triple and rotates on refresh / revocation.
"""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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

    async def cleanup_expired(self, *, before: float | None = None) -> int:
        """Delete every row whose ``expires_at`` has passed.

        Default implementation is a no-op so existing custom subclasses
        keep working unchanged — concrete SQL-backed stores override
        with a single ``DELETE`` statement. Returns the number of rows
        actually deleted so callers can log the figure for capacity
        planning.

        ``before`` is the cut-off epoch seconds; defaults to the
        current wall-clock time. Tokens with ``expires_at == 0`` (no
        expiry info) are never deleted by this method — call sites
        that want to purge stale-no-expiry rows should add their own
        TTL.
        """
        return 0
