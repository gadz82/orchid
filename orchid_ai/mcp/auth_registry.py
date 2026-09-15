"""
Immutable registry of MCP servers that require per-server OAuth.

Under the MCP 2025-03-26 authorization spec, the framework no longer
needs (or accepts) static ``client_id`` / ``authorization_endpoint`` /
``token_endpoint`` values in YAML — those are discovered at runtime
from the server's 401 response (see :mod:`orchid_ai.mcp.discovery`).
The registry therefore only tracks **which** servers are declared
``auth.mode: oauth`` and which agents depend on them; all other
metadata comes from :class:`OrchidMCPClientRegistrationStore` once
discovery has run.

Built once from :class:`~orchid_ai.config.schema.OrchidAgentsConfig`
at graph startup via :meth:`OrchidMCPAuthRegistry.from_config`.  When
the same server name appears in multiple agents their ``agent_names``
lists are merged into a single tuple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import OrchidAgentsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchidMCPOAuthServerInfo:
    """Identity of a per-server OAuth MCP server.

    Intentionally narrow: the framework only needs the server name, the
    MCP endpoint URL (so the API layer can probe it for the
    ``WWW-Authenticate`` header on first Connect), and the list of
    agents that depend on it.  Everything else (authorization-server
    endpoints, client credentials, scopes) is discovered at runtime and
    persisted in :class:`OrchidMCPClientRegistrationStore`.
    """

    server_name: str
    url: str
    agent_names: tuple[str, ...]  # frozen → must be tuple, not list
    #: Manual OAuth config seeded from YAML (authorization_endpoint,
    #: token_endpoint, client_id, client_secret, scopes, issuer) for
    #: servers that are NOT MCP 2025-03-26 compliant.  ``None`` means
    #: the framework must run the standard auto-discovery chain.
    manual_oauth_config: dict | None = None


@dataclass
class OrchidMCPAuthRegistry:
    """Immutable registry of OAuth-requiring MCP servers.

    Constructed via the :meth:`from_config` class method — never
    instantiated directly by consumers.
    """

    _servers: dict[str, OrchidMCPOAuthServerInfo] = field(default_factory=dict)

    # ── Public API ────────────────────────────────────────────

    @property
    def oauth_servers(self) -> dict[str, OrchidMCPOAuthServerInfo]:
        """All OAuth servers keyed by name (read-only view)."""
        return dict(self._servers)

    @property
    def empty(self) -> bool:
        """True when no OAuth servers are registered."""
        return len(self._servers) == 0

    def get_server(self, name: str) -> OrchidMCPOAuthServerInfo | None:
        """Retrieve info for a specific server, or ``None``."""
        return self._servers.get(name)

    def requires_oauth(self, name: str) -> bool:
        """True when the named server requires per-user OAuth."""
        return name in self._servers

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: OrchidAgentsConfig) -> OrchidMCPAuthRegistry:
        """Scan all agents and collect OAuth-requiring MCP servers.

        When the same ``server_name`` appears in multiple agents their
        ``agent_names`` are merged into a single tuple.  Child-agent
        names are qualified as ``"{parent}.{child}"`` so the same server
        referenced by both a parent and a child yields two entries in
        the merged list.
        """
        # server_name → {"url": str, "agents": list[str]}
        server_map: dict[str, dict] = {}

        def _collect(mcp_servers, agent_ref: str) -> None:
            for server in mcp_servers:
                if server.auth.mode != "oauth":
                    continue
                entry = server_map.setdefault(
                    server.name,
                    {"url": server.url, "agents": [], "manual_oauth_config": None},
                )
                if agent_ref not in entry["agents"]:
                    entry["agents"].append(agent_ref)
                # First URL seen wins; a later duplicate with a different
                # URL is a config mistake we surface once, loudly.
                if entry["url"] != server.url:
                    logger.warning(
                        "[OrchidMCPAuthRegistry] Server '%s' declared with "
                        "conflicting URLs (%r vs %r) — using the first",
                        server.name,
                        entry["url"],
                        server.url,
                    )
                # Carry manual OAuth config from YAML (for non-compliant servers)
                manual = server.auth.manual_registration
                if manual and manual.authorization_endpoint and manual.token_endpoint:
                    entry["manual_oauth_config"] = {
                        "authorization_endpoint": manual.authorization_endpoint,
                        "token_endpoint": manual.token_endpoint,
                        "registration_endpoint": manual.registration_endpoint,
                        "client_id": manual.client_id,
                        "client_secret": manual.client_secret,
                        "scopes": manual.scopes,
                        "issuer": manual.issuer,
                    }

        for agent_name, agent_config in config.agents.items():
            _collect(agent_config.mcp_servers, agent_name)
            if agent_config.children:
                for child_name, child_config in agent_config.children.items():
                    _collect(child_config.mcp_servers, f"{agent_name}.{child_name}")

        servers = {
            name: OrchidMCPOAuthServerInfo(
                server_name=name,
                url=entry["url"],
                agent_names=tuple(entry["agents"]),
                manual_oauth_config=entry.get("manual_oauth_config"),
            )
            for name, entry in server_map.items()
        }

        if servers:
            logger.info(
                "[OrchidMCPAuthRegistry] %d OAuth server(s): %s",
                len(servers),
                ", ".join(f"{n} (agents: {', '.join(s.agent_names)})" for n, s in servers.items()),
            )
        else:
            logger.debug("[OrchidMCPAuthRegistry] No OAuth servers configured")

        return cls(_servers=servers)
