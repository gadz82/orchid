"""Tests for OrchidMCPServerInventory.

Asserts the inventory-derivation contract used by
:class:`OrchidSessionWarmer`: deduping by ``(server_name, url)``,
qualifying child agent references, mode filtering, and the
``clients_for`` lookup that finds the right per-agent client instances.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidMCPAuthConfig,
    OrchidMCPServerConfig,
)
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.inventory import OrchidMCPServerEntry, OrchidMCPServerInventory

# ── Helpers ────────────────────────────────────────────────────


def _server(name: str, url: str | None = None, mode: str = "none") -> OrchidMCPServerConfig:
    return OrchidMCPServerConfig(
        name=name,
        url=url or f"https://{name}.example.com/mcp",
        auth=OrchidMCPAuthConfig(mode=mode),
    )


def _agent(*servers: OrchidMCPServerConfig, **extra: Any) -> OrchidAgentConfig:
    return OrchidAgentConfig(
        description="t",
        prompt="t",
        mcp_servers=list(servers),
        **extra,
    )


def _config(agents: dict[str, OrchidAgentConfig]) -> OrchidAgentsConfig:
    return OrchidAgentsConfig(agents=agents)


class _StubAgent:
    """Tiny duck-typed stand-in for ``GenericAgent`` — only what the inventory needs."""

    def __init__(self, agent_config: OrchidAgentConfig, mcp_clients: list[OrchidMCPClient]) -> None:
        self._config = agent_config
        self.mcp_clients = mcp_clients


class _StubClient(OrchidMCPClient):
    def __init__(self, url: str) -> None:
        self._url = url
        self.warm_calls = 0

    @property
    def server_url(self) -> str:
        return self._url

    async def call_tool(self, tool_name, arguments, auth):  # pragma: no cover
        return OrchidMCPToolResult()

    async def list_tools(self, auth):  # pragma: no cover
        return []

    async def list_prompts(self, auth):  # pragma: no cover
        return []

    async def list_resources(self, auth):  # pragma: no cover
        return []

    async def get_prompt(self, name, arguments, auth):  # pragma: no cover
        return []

    async def read_resource(self, uri, auth):  # pragma: no cover
        return ""

    async def warm_cache(self, auth: OrchidAuthContext) -> None:  # pragma: no cover
        self.warm_calls += 1


# ── Tests ──────────────────────────────────────────────────────


class TestFromConfig:
    def test_empty_config_yields_empty_inventory(self):
        inv = OrchidMCPServerInventory.from_config(_config({}))
        assert inv.empty
        assert inv.entries() == []

    def test_collects_all_modes(self):
        cfg = _config(
            {
                "alpha": _agent(_server("local-tool", mode="none")),
                "beta": _agent(_server("internal-rest", mode="passthrough")),
                "gamma": _agent(_server("ext-crm", mode="oauth")),
            }
        )
        inv = OrchidMCPServerInventory.from_config(cfg)
        modes = {e.server_name: e.mode for e in inv.entries()}
        assert modes == {"local-tool": "none", "internal-rest": "passthrough", "ext-crm": "oauth"}

    def test_dedupes_same_name_and_url(self):
        srv = _server("shared", mode="passthrough")
        cfg = _config({"a": _agent(srv), "b": _agent(srv)})
        inv = OrchidMCPServerInventory.from_config(cfg)
        entries = inv.entries()
        assert len(entries) == 1
        assert set(entries[0].agent_names) == {"a", "b"}

    def test_qualifies_child_agent_names(self):
        cfg = _config(
            {
                "parent": OrchidAgentConfig(
                    description="t",
                    prompt="t",
                    mcp_servers=[],
                    children={
                        "kid-a": OrchidAgentConfig(
                            description="t",
                            prompt="t",
                            mcp_servers=[_server("ext", mode="oauth")],
                        ),
                    },
                ),
            }
        )
        inv = OrchidMCPServerInventory.from_config(cfg)
        entries = inv.entries()
        assert len(entries) == 1
        assert entries[0].agent_names == ("parent.kid-a",)

    def test_warns_on_conflicting_mode(self, caplog):
        # Same (name, url) appearing twice with different modes — the
        # first one wins; we want the warning logged once.
        srv_a = _server("dual", mode="passthrough")
        srv_b = OrchidMCPServerConfig(
            name="dual",
            url=srv_a.url,
            auth=OrchidMCPAuthConfig(mode="oauth"),
        )
        cfg = _config({"a": _agent(srv_a), "b": _agent(srv_b)})
        with caplog.at_level(logging.WARNING, logger="orchid_ai.mcp.inventory"):
            inv = OrchidMCPServerInventory.from_config(cfg)
        # First-seen mode wins.
        entry = inv.entries()[0]
        assert entry.mode == "passthrough"
        assert any("conflicting auth modes" in m for m in caplog.messages)

    def test_keeps_separate_entries_for_same_name_different_url(self):
        srv_a = _server("dup", url="https://a.example.com/mcp", mode="none")
        srv_b = _server("dup", url="https://b.example.com/mcp", mode="none")
        cfg = _config({"x": _agent(srv_a), "y": _agent(srv_b)})
        inv = OrchidMCPServerInventory.from_config(cfg)
        # Two entries with the same name but different URLs — both kept.
        urls = sorted(e.url for e in inv.entries())
        assert urls == ["https://a.example.com/mcp", "https://b.example.com/mcp"]


class TestEntriesWithMode:
    def test_filters_by_mode(self):
        cfg = _config(
            {
                "a": _agent(
                    _server("local-tool", mode="none"),
                    _server("internal", mode="passthrough"),
                    _server("ext", mode="oauth"),
                ),
            }
        )
        inv = OrchidMCPServerInventory.from_config(cfg)
        none_names = [e.server_name for e in inv.entries_with_mode("none")]
        oauth_names = [e.server_name for e in inv.entries_with_mode("oauth")]
        passthrough_names = [e.server_name for e in inv.entries_with_mode("passthrough")]
        assert none_names == ["local-tool"]
        assert oauth_names == ["ext"]
        assert passthrough_names == ["internal"]


class TestClientsFor:
    def test_returns_positional_match(self):
        srv = _server("api", mode="passthrough")
        agent_config = _agent(srv)
        client = _StubClient(srv.url)
        agents = {"acct": _StubAgent(agent_config, [client])}

        cfg = _config({"acct": agent_config})
        inv = OrchidMCPServerInventory.from_config(cfg)
        entry = inv.entries()[0]

        clients = inv.clients_for(entry, agents)
        assert clients == [client]

    def test_skips_child_qualified_names(self):
        # A server that lives entirely inside a child — top-level agents
        # have no client for it, so clients_for returns an empty list.
        cfg = _config(
            {
                "parent": OrchidAgentConfig(
                    description="t",
                    prompt="t",
                    mcp_servers=[],
                    children={
                        "kid": OrchidAgentConfig(
                            description="t",
                            prompt="t",
                            mcp_servers=[_server("kid-only", mode="none")],
                        ),
                    },
                ),
            }
        )
        inv = OrchidMCPServerInventory.from_config(cfg)
        entry = inv.entries()[0]
        # Empty top-level agents dict — child-only entries are unreachable.
        assert inv.clients_for(entry, {}) == []

    def test_dedupes_clients_across_agents(self):
        # When two agents share the SAME client instance, clients_for
        # should not return it twice.
        srv = _server("shared", mode="none")
        agent_a_cfg = _agent(srv)
        agent_b_cfg = _agent(srv)
        shared_client = _StubClient(srv.url)
        agents = {
            "a": _StubAgent(agent_a_cfg, [shared_client]),
            "b": _StubAgent(agent_b_cfg, [shared_client]),
        }

        cfg = _config({"a": agent_a_cfg, "b": agent_b_cfg})
        inv = OrchidMCPServerInventory.from_config(cfg)
        entry = inv.entries()[0]

        clients = inv.clients_for(entry, agents)
        assert clients == [shared_client]

    def test_returns_distinct_per_agent_instances(self):
        srv = _server("api", mode="passthrough")
        agent_a_cfg = _agent(srv)
        agent_b_cfg = _agent(srv)
        client_a = _StubClient(srv.url)
        client_b = _StubClient(srv.url)
        agents = {
            "a": _StubAgent(agent_a_cfg, [client_a]),
            "b": _StubAgent(agent_b_cfg, [client_b]),
        }
        cfg = _config({"a": agent_a_cfg, "b": agent_b_cfg})
        inv = OrchidMCPServerInventory.from_config(cfg)
        entry = inv.entries()[0]

        clients = inv.clients_for(entry, agents)
        assert set(map(id, clients)) == {id(client_a), id(client_b)}

    def test_skips_agent_without_clients(self):
        # Agent registered in config but ``mcp_clients`` empty (e.g. a
        # custom agent that ignored its injected list).
        srv = _server("api", mode="none")
        agent_cfg = _agent(srv)
        agents = {"x": _StubAgent(agent_cfg, [])}

        cfg = _config({"x": agent_cfg})
        inv = OrchidMCPServerInventory.from_config(cfg)
        entry = inv.entries()[0]
        assert inv.clients_for(entry, agents) == []

    def test_url_fallback_for_custom_agent_without_config(self):
        srv_a = _server("a", url="https://aa.example.com/mcp", mode="none")
        srv_b = _server("b", url="https://bb.example.com/mcp", mode="none")
        client_a = _StubClient(srv_a.url)
        client_b = _StubClient(srv_b.url)

        # Custom agent with no ``_config`` attribute — URL match falls
        # back to ``server_url``.
        class _CustomAgent:
            def __init__(self):
                self.mcp_clients = [client_a, client_b]

        agent_cfg = _agent(srv_a, srv_b)
        agents = {"custom": _CustomAgent()}
        # Provide the entry through the config, not through the agent's
        # config attribute — the inventory still has it.
        cfg = _config({"custom": agent_cfg})
        inv = OrchidMCPServerInventory.from_config(cfg)

        entry_a = next(e for e in inv.entries() if e.server_name == "a")
        clients = inv.clients_for(entry_a, agents)
        assert clients == [client_a]


class TestEntryFrozen:
    def test_entry_is_frozen(self):
        entry = OrchidMCPServerEntry(
            server_name="x",
            url="https://x/mcp",
            mode="none",
            agent_names=("a",),
        )
        with pytest.raises(FrozenInstanceError):
            entry.server_name = "y"  # type: ignore[misc]
