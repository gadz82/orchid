"""Tests for OrchidMCPAuthRegistry — scans OrchidAgentsConfig for OAuth-requiring MCP servers.

After the MCP 2025-03-26 spec migration the registry is a thin
"which agents depend on which OAuth servers?" mapping — endpoints and
credentials are discovered at runtime.  These tests lock that contract.
"""

from __future__ import annotations

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidMCPAuthConfig,
    OrchidMCPServerConfig,
)
from orchid_ai.mcp.auth_registry import OrchidMCPAuthRegistry


def _make_config(agents: dict) -> OrchidAgentsConfig:
    """Build a minimal OrchidAgentsConfig with the given agents dict."""
    return OrchidAgentsConfig(agents=agents)


def _oauth_server(name: str) -> OrchidMCPServerConfig:
    return OrchidMCPServerConfig(
        name=name,
        url=f"https://{name}.example.com/mcp",
        auth=OrchidMCPAuthConfig(mode="oauth"),
    )


class TestMCPAuthRegistry:
    def test_empty_when_no_oauth_servers(self):
        config = _make_config(
            {
                "agent1": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    mcp_servers=[
                        OrchidMCPServerConfig(name="local", url="http://localhost:3000/mcp"),
                        OrchidMCPServerConfig(
                            name="passthrough-server",
                            url="http://internal/mcp",
                            auth=OrchidMCPAuthConfig(mode="passthrough"),
                        ),
                    ],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert registry.empty
        assert registry.oauth_servers == {}

    def test_discovers_oauth_servers(self):
        config = _make_config(
            {
                "sales": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    mcp_servers=[
                        OrchidMCPServerConfig(name="internal", url="http://localhost/mcp"),
                        _oauth_server("ext-crm"),
                    ],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert not registry.empty
        assert "ext-crm" in registry.oauth_servers
        info = registry.get_server("ext-crm")
        assert info is not None
        assert info.server_name == "ext-crm"
        assert info.agent_names == ("sales",)

    def test_info_contains_no_static_credentials(self):
        """Registry info is intentionally minimal — credentials live in the discovery store."""
        config = _make_config(
            {
                "sales": OrchidAgentConfig(
                    description="t",
                    prompt="t",
                    mcp_servers=[_oauth_server("ext-crm")],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        info = registry.get_server("ext-crm")
        assert info is not None
        for legacy in (
            "client_id",
            "client_secret",
            "authorization_endpoint",
            "token_endpoint",
            "scopes",
            "issuer",
        ):
            assert not hasattr(info, legacy), (
                f"registry info leaked legacy field '{legacy}' — belongs in the "
                f"OrchidMCPClientRegistrationStore, not the registry"
            )

    def test_merges_agent_names_for_same_server(self):
        """Same MCP server in multiple agents → merged agent_names."""
        config = _make_config(
            {
                "sales": OrchidAgentConfig(
                    description="sales",
                    prompt="t",
                    mcp_servers=[_oauth_server("ext-crm")],
                ),
                "support": OrchidAgentConfig(
                    description="support",
                    prompt="t",
                    mcp_servers=[_oauth_server("ext-crm")],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        info = registry.get_server("ext-crm")
        assert info is not None
        assert set(info.agent_names) == {"sales", "support"}

    def test_requires_oauth(self):
        config = _make_config(
            {
                "agent1": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    mcp_servers=[_oauth_server("ext")],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert registry.requires_oauth("ext") is True
        assert registry.requires_oauth("nonexistent") is False

    def test_get_server_returns_none_for_unknown(self):
        config = _make_config({})
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert registry.get_server("anything") is None

    def test_ignores_none_mode_servers(self):
        """Servers with mode=none (default) should not appear in registry."""
        config = _make_config(
            {
                "agent1": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    mcp_servers=[OrchidMCPServerConfig(name="local", url="http://localhost/mcp")],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert registry.empty

    def test_child_agent_names_qualified(self):
        """Child-agent references are prefixed ``{parent}.{child}``."""
        parent = OrchidAgentConfig(
            description="parent",
            prompt="t",
            mcp_servers=[],
            children={
                "child-a": OrchidAgentConfig(
                    description="child",
                    prompt="t",
                    mcp_servers=[_oauth_server("ext")],
                ),
            },
        )
        config = _make_config({"parent": parent})
        registry = OrchidMCPAuthRegistry.from_config(config)
        info = registry.get_server("ext")
        assert info is not None
        assert info.agent_names == ("parent.child-a",)
