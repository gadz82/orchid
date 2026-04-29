"""
Immutable inventory of every MCP server declared in ``OrchidAgentsConfig``.

Where :class:`OrchidMCPAuthRegistry` only tracks ``auth.mode: oauth``
servers (so the supervisor can render an auth-aware routing prompt),
:class:`OrchidMCPServerInventory` carries **every** declared server —
``none`` and ``passthrough`` included — so the
:class:`OrchidSessionWarmer` can warm capability caches at the right
lifecycle boundary (process startup vs. user-session start).

Built once from the parsed config; never mutated.  Servers are deduped
by ``(server_name, url)`` and the list of agents that reference each
entry is merged.  Child-agent references are qualified
``"{parent}.{child}"`` so a single server reachable through both a
parent and a child surfaces both qualifiers in ``agent_names`` — the
shape mirrors :class:`OrchidMCPAuthRegistry.from_config` so the two
registries stay easy to reason about side-by-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..config.schema import OrchidAgentsConfig
    from ..core.agent import OrchidAgent
    from ..core.mcp import OrchidMCPClient

logger = logging.getLogger(__name__)


OrchidMCPAuthMode = Literal["none", "passthrough", "oauth"]


@dataclass(frozen=True)
class OrchidMCPServerEntry:
    """One unique MCP server declared anywhere in the config.

    Two server configs collapse into the same entry when they share
    ``server_name`` AND ``url``.  ``agent_names`` collects every agent
    qualifier (top-level name or ``parent.child``) referencing the
    entry, in declaration order.
    """

    server_name: str
    url: str
    mode: OrchidMCPAuthMode
    agent_names: tuple[str, ...]


class OrchidMCPServerInventory:
    """Read-only view of every MCP server declared in the loaded config.

    Construct via :meth:`from_config`; the constructor is for the
    factory's own use.
    """

    def __init__(self, entries: dict[tuple[str, str], OrchidMCPServerEntry]) -> None:
        self._entries = entries

    # ── Public API ────────────────────────────────────────────

    def entries(self) -> list[OrchidMCPServerEntry]:
        """Every (server_name, url) entry in declaration order."""
        return list(self._entries.values())

    def entries_with_mode(self, mode: OrchidMCPAuthMode) -> list[OrchidMCPServerEntry]:
        """Entries whose ``auth.mode`` equals ``mode``."""
        return [entry for entry in self._entries.values() if entry.mode == mode]

    @property
    def empty(self) -> bool:
        """True when no MCP servers are declared anywhere in the config."""
        return not self._entries

    def clients_for(
        self,
        entry: OrchidMCPServerEntry,
        agents: dict[str, OrchidAgent],
    ) -> list[OrchidMCPClient]:
        """Collect every per-agent ``OrchidMCPClient`` bound to ``entry``.

        Walks the top-level ``agents`` dict only — child agents live
        inside compiled LangGraph subgraphs so their per-agent client
        instances are not directly reachable.  Their first-call
        discovery still works lazily; the warmer accepts that trade-off
        as the simpler "warm-on-every-top-level-instance" cut documented
        in ``.knowledge/mcp-startup-discovery-plan.md`` §9.

        Defensive: an agent might not actually have a client at the
        expected index (mismatched config, custom agent that ignored its
        ``mcp_clients`` list), in which case we skip it.
        """
        matched: list[OrchidMCPClient] = []
        seen: set[int] = set()
        for agent_name in entry.agent_names:
            # Child references qualify as ``parent.child`` and are not
            # present in the top-level agents dict — see docstring.
            if "." in agent_name:
                continue
            agent = agents.get(agent_name)
            if agent is None:
                continue

            agent_clients = getattr(agent, "mcp_clients", None) or []
            agent_config = getattr(agent, "_config", None)

            # Preferred: positional match using the agent's own config —
            # an agent's ``mcp_clients[i]`` is built from
            # ``mcp_servers[i]`` (graph.py:_instantiate_agent), so the
            # index gives us an exact instance.
            if agent_config is not None and getattr(agent_config, "mcp_servers", None):
                for i, server in enumerate(agent_config.mcp_servers):
                    if server.name == entry.server_name and server.url == entry.url and i < len(agent_clients):
                        client = agent_clients[i]
                        client_id = id(client)
                        if client_id not in seen:
                            matched.append(client)
                            seen.add(client_id)
                continue

            # Fallback for custom agents without ``_config``: match by
            # ``server_url``.  We cannot disambiguate by name in that
            # case, but two clients with the same URL are indistinguish-
            # able to the warmer anyway — both share the same upstream.
            for client in agent_clients:
                if getattr(client, "server_url", None) == entry.url:
                    client_id = id(client)
                    if client_id not in seen:
                        matched.append(client)
                        seen.add(client_id)

        return matched

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: OrchidAgentsConfig) -> OrchidMCPServerInventory:
        """Scan every agent (and child) and build a deduped inventory.

        Mirrors :meth:`OrchidMCPAuthRegistry.from_config` for shape and
        warning behaviour; differs only in that it includes ``none`` and
        ``passthrough`` servers as well.
        """
        entries: dict[tuple[str, str], OrchidMCPServerEntry] = {}
        # Track first-seen mode per (name, url) so we can warn on conflict.
        seen_modes: dict[tuple[str, str], str] = {}
        # Mutable working storage for agent_names that we freeze at the end.
        agent_lists: dict[tuple[str, str], list[str]] = {}

        def _collect(mcp_servers, agent_ref: str) -> None:
            for server in mcp_servers:
                key = (server.name, server.url)
                mode: OrchidMCPAuthMode = server.auth.mode
                if key not in entries:
                    entries[key] = OrchidMCPServerEntry(
                        server_name=server.name,
                        url=server.url,
                        mode=mode,
                        agent_names=(),
                    )
                    seen_modes[key] = mode
                    agent_lists[key] = []
                else:
                    if seen_modes[key] != mode:
                        logger.warning(
                            "[OrchidMCPServerInventory] Server '%s' at %s declared with "
                            "conflicting auth modes (%r vs %r) — using the first",
                            server.name,
                            server.url,
                            seen_modes[key],
                            mode,
                        )
                if agent_ref not in agent_lists[key]:
                    agent_lists[key].append(agent_ref)

        for agent_name, agent_config in config.agents.items():
            _collect(agent_config.mcp_servers, agent_name)
            if agent_config.children:
                for child_name, child_config in agent_config.children.items():
                    _collect(child_config.mcp_servers, f"{agent_name}.{child_name}")

        # Warn on (server_name, _) conflict — same name, different URLs.
        url_by_name: dict[str, str] = {}
        for name, url in entries:
            if name in url_by_name and url_by_name[name] != url:
                logger.warning(
                    "[OrchidMCPServerInventory] Server name '%s' declared with "
                    "conflicting URLs (%r vs %r) — both kept as separate entries",
                    name,
                    url_by_name[name],
                    url,
                )
            else:
                url_by_name.setdefault(name, url)

        # Re-materialise entries with frozen agent_names tuples.
        finalised = {
            key: OrchidMCPServerEntry(
                server_name=entry.server_name,
                url=entry.url,
                mode=entry.mode,
                agent_names=tuple(agent_lists[key]),
            )
            for key, entry in entries.items()
        }

        if finalised:
            logger.debug(
                "[OrchidMCPServerInventory] %d MCP server(s) discovered: %s",
                len(finalised),
                ", ".join(f"{e.server_name}({e.mode})" for e in finalised.values()),
            )
        else:
            logger.debug("[OrchidMCPServerInventory] No MCP servers configured")

        return cls(finalised)
