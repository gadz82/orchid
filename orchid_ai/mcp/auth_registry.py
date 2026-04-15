"""
Immutable registry of MCP servers that require per-server OAuth.

Built once from ``AgentsConfig`` at graph startup via
``MCPAuthRegistry.from_config(config)``.  Scans all agents' MCP
servers and collects those with ``auth.mode == "oauth"``.  When the
same server name appears in multiple agents their ``agent_names``
lists are merged.

The registry is stored on ``OrchidRuntime.mcp_auth_registry`` and
exposed to the API layer for authorization-status endpoints and
pre-flight auth checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import AgentsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPOAuthServerInfo:
    """Static OAuth metadata for a single MCP server."""

    server_name: str
    client_id: str
    authorization_endpoint: str
    token_endpoint: str
    scopes: str
    issuer: str
    agent_names: tuple[str, ...]  # frozen → must be tuple, not list


@dataclass
class MCPAuthRegistry:
    """Immutable registry of OAuth-requiring MCP servers.

    Constructed via the :meth:`from_config` class method — never
    instantiated directly by consumers.
    """

    _servers: dict[str, MCPOAuthServerInfo] = field(default_factory=dict)

    # ── Public API ────────────────────────────────────────────

    @property
    def oauth_servers(self) -> dict[str, MCPOAuthServerInfo]:
        """All OAuth servers keyed by name (read-only view)."""
        return dict(self._servers)

    @property
    def empty(self) -> bool:
        """True when no OAuth servers are registered."""
        return len(self._servers) == 0

    def get_server(self, name: str) -> MCPOAuthServerInfo | None:
        """Retrieve info for a specific server, or ``None``."""
        return self._servers.get(name)

    def requires_oauth(self, name: str) -> bool:
        """True when the named server requires per-user OAuth."""
        return name in self._servers

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: AgentsConfig) -> MCPAuthRegistry:
        """Scan all agents and collect OAuth-requiring MCP servers.

        When the same ``server_name`` appears in multiple agents their
        ``agent_names`` are merged into a single tuple.
        """
        server_map: dict[str, dict] = {}  # server_name → builder dict

        for agent_name, agent_config in config.agents.items():
            for mcp_server in agent_config.mcp_servers:
                if mcp_server.auth.mode != "oauth":
                    continue

                name = mcp_server.name
                if name in server_map:
                    # Merge agent names
                    existing = server_map[name]
                    if agent_name not in existing["agent_names"]:
                        existing["agent_names"].append(agent_name)
                else:
                    server_map[name] = {
                        "server_name": name,
                        "client_id": mcp_server.auth.client_id,
                        "authorization_endpoint": mcp_server.auth.authorization_endpoint,
                        "token_endpoint": mcp_server.auth.token_endpoint,
                        "scopes": mcp_server.auth.scopes,
                        "issuer": mcp_server.auth.issuer,
                        "agent_names": [agent_name],
                    }

            # Recurse into children if present
            if agent_config.children:
                for child_name, child_config in agent_config.children.items():
                    for mcp_server in child_config.mcp_servers:
                        if mcp_server.auth.mode != "oauth":
                            continue

                        name = mcp_server.name
                        qualified_name = f"{agent_name}.{child_name}"
                        if name in server_map:
                            existing = server_map[name]
                            if qualified_name not in existing["agent_names"]:
                                existing["agent_names"].append(qualified_name)
                        else:
                            server_map[name] = {
                                "server_name": name,
                                "client_id": mcp_server.auth.client_id,
                                "authorization_endpoint": mcp_server.auth.authorization_endpoint,
                                "token_endpoint": mcp_server.auth.token_endpoint,
                                "scopes": mcp_server.auth.scopes,
                                "issuer": mcp_server.auth.issuer,
                                "agent_names": [qualified_name],
                            }

        # Build frozen dataclass instances
        servers = {
            name: MCPOAuthServerInfo(
                server_name=data["server_name"],
                client_id=data["client_id"],
                authorization_endpoint=data["authorization_endpoint"],
                token_endpoint=data["token_endpoint"],
                scopes=data["scopes"],
                issuer=data["issuer"],
                agent_names=tuple(data["agent_names"]),
            )
            for name, data in server_map.items()
        }

        if servers:
            logger.info(
                "[MCPAuthRegistry] %d OAuth server(s): %s",
                len(servers),
                ", ".join(f"{n} (agents: {', '.join(s.agent_names)})" for n, s in servers.items()),
            )
        else:
            logger.debug("[MCPAuthRegistry] No OAuth servers configured")

        return cls(_servers=servers)
