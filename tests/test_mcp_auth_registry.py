"""Tests for OrchidMCPAuthRegistry — scans OrchidAgentsConfig for OAuth-requiring MCP servers."""

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
                        OrchidMCPServerConfig(
                            name="ext-crm",
                            url="https://crm.example.com/mcp",
                            auth=OrchidMCPAuthConfig(
                                mode="oauth",
                                client_id="orchid-crm",
                                issuer="https://auth.crm.example.com",
                                scopes="openid crm.read",
                            ),
                        ),
                    ],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert not registry.empty
        assert "ext-crm" in registry.oauth_servers
        info = registry.get_server("ext-crm")
        assert info is not None
        assert info.client_id == "orchid-crm"
        assert info.agent_names == ("sales",)

    def test_merges_agent_names_for_same_server(self):
        """Same MCP server in multiple agents → merged agent_names."""
        crm_auth = OrchidMCPAuthConfig(mode="oauth", client_id="crm", issuer="https://auth.crm.com")
        config = _make_config(
            {
                "sales": OrchidAgentConfig(
                    description="sales",
                    prompt="test",
                    mcp_servers=[
                        OrchidMCPServerConfig(name="ext-crm", url="https://crm/mcp", auth=crm_auth),
                    ],
                ),
                "support": OrchidAgentConfig(
                    description="support",
                    prompt="test",
                    mcp_servers=[
                        OrchidMCPServerConfig(name="ext-crm", url="https://crm/mcp", auth=crm_auth),
                    ],
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
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="ext",
                            url="https://ext/mcp",
                            auth=OrchidMCPAuthConfig(mode="oauth", client_id="x"),
                        ),
                    ],
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
                    mcp_servers=[
                        OrchidMCPServerConfig(name="local", url="http://localhost/mcp"),
                    ],
                ),
            }
        )
        registry = OrchidMCPAuthRegistry.from_config(config)
        assert registry.empty
