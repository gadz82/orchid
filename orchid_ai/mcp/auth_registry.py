"""
Immutable registry of MCP servers that require per-server OAuth.

Built once from ``OrchidAgentsConfig`` at graph startup via
``OrchidMCPAuthRegistry.from_config(config)``.  Scans all agents' MCP
servers and collects those with ``auth.mode == "oauth"``.  When the
same server name appears in multiple agents their ``agent_names``
lists are merged.

The registry is stored on ``OrchidRuntime.mcp_auth_registry`` and
exposed to the API layer for authorization-status endpoints and
pre-flight auth checks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import OrchidAgentsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchidMCPOAuthServerInfo:
    """Static OAuth metadata for a single MCP server.

    ``client_secret`` is resolved from the ``client_secret_env``
    environment variable at registry-build time.  An empty string means
    "public client" (PKCE-only) — confidential clients must set the env
    var before the API server starts.
    """

    server_name: str
    client_id: str
    authorization_endpoint: str
    token_endpoint: str
    scopes: str
    issuer: str
    agent_names: tuple[str, ...]  # frozen → must be tuple, not list
    client_secret: str = ""  # resolved from client_secret_env at build time


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
        ``agent_names`` are merged into a single tuple.
        """
        server_map: dict[str, dict] = {}  # server_name → builder dict

        def _build_entry(mcp_server, agent_ref: str) -> dict:
            """Turn an ``OrchidMCPServerConfig`` into a registry builder dict.

            ``client_secret`` is resolved once here from the
            ``client_secret_env`` name so downstream code never has to
            touch ``os.environ`` again and tests can monkeypatch the env
            before calling ``from_config``.
            """
            client_secret = ""
            if mcp_server.auth.client_secret_env:
                client_secret = os.environ.get(mcp_server.auth.client_secret_env, "")
                if not client_secret:
                    logger.warning(
                        "[OrchidMCPAuthRegistry] client_secret_env=%r for server '%s' "
                        "is unset in the process environment — token exchange will "
                        "be attempted without a client secret",
                        mcp_server.auth.client_secret_env,
                        mcp_server.name,
                    )
            return {
                "server_name": mcp_server.name,
                "client_id": mcp_server.auth.client_id,
                "client_secret": client_secret,
                "authorization_endpoint": mcp_server.auth.authorization_endpoint,
                "token_endpoint": mcp_server.auth.token_endpoint,
                "scopes": mcp_server.auth.scopes,
                "issuer": mcp_server.auth.issuer,
                "agent_names": [agent_ref],
            }

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
                    server_map[name] = _build_entry(mcp_server, agent_name)

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
                            server_map[name] = _build_entry(mcp_server, qualified_name)

        # Build frozen dataclass instances
        servers = {
            name: OrchidMCPOAuthServerInfo(
                server_name=data["server_name"],
                client_id=data["client_id"],
                client_secret=data["client_secret"],
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
                "[OrchidMCPAuthRegistry] %d OAuth server(s): %s",
                len(servers),
                ", ".join(f"{n} (agents: {', '.join(s.agent_names)})" for n, s in servers.items()),
            )
        else:
            logger.debug("[OrchidMCPAuthRegistry] No OAuth servers configured")

        return cls(_servers=servers)
