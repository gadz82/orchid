"""Tests for Human-in-the-Loop tool approval (#7)."""

from __future__ import annotations

from unittest.mock import MagicMock


from orchid_ai.config.schema import (
    AgentConfig,
    AgentsConfig,
    BuiltinToolConfig,
    MCPServerConfig,
    RAGConfig,
    ToolConfig,
)


# ── Schema tests: requires_approval field ───────────────────


class TestToolConfigApproval:
    """ToolConfig.requires_approval field (MCP tools)."""

    def test_default_no_approval(self):
        cfg = ToolConfig(name="list_items")
        assert cfg.requires_approval is False

    def test_approval_set(self):
        cfg = ToolConfig(name="create_item", requires_approval=True)
        assert cfg.requires_approval is True

    def test_from_dict(self):
        data = {"name": "delete_item", "requires_approval": True}
        cfg = ToolConfig(**data)
        assert cfg.requires_approval is True


class TestBuiltinToolConfigApproval:
    """BuiltinToolConfig.requires_approval field (built-in tools)."""

    def test_default_no_approval(self):
        cfg = BuiltinToolConfig(handler="mymod.fn")
        assert cfg.requires_approval is False

    def test_approval_set(self):
        cfg = BuiltinToolConfig(handler="mymod.fn", requires_approval=True)
        assert cfg.requires_approval is True


class TestApprovalToolsComputed:
    """AgentConfig.approval_tools computed from MCP + built-in configs."""

    def test_mcp_tools_collected(self):
        config = AgentsConfig(
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                    mcp_servers=[
                        MCPServerConfig(
                            name="api",
                            url="http://localhost:8080",
                            tools=[
                                ToolConfig(name="list_items"),
                                ToolConfig(name="create_item", requires_approval=True),
                                ToolConfig(name="delete_item", requires_approval=True),
                            ],
                        ),
                    ],
                ),
            },
        )
        agent = config.agents["test"]
        assert agent.approval_tools == {"create_item", "delete_item"}

    def test_builtin_tools_collected(self):
        config = AgentsConfig(
            tools={
                "search": BuiltinToolConfig(handler="mymod.search"),
                "send_email": BuiltinToolConfig(handler="mymod.email", requires_approval=True),
            },
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                    tools=["search", "send_email"],
                ),
            },
        )
        agent = config.agents["test"]
        assert "send_email" in agent.approval_tools
        assert "search" not in agent.approval_tools

    def test_mixed_mcp_and_builtin(self):
        config = AgentsConfig(
            tools={
                "dangerous_fn": BuiltinToolConfig(handler="mymod.danger", requires_approval=True),
            },
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                    tools=["dangerous_fn"],
                    mcp_servers=[
                        MCPServerConfig(
                            name="api",
                            url="http://localhost:8080",
                            tools=[
                                ToolConfig(name="write_data", requires_approval=True),
                            ],
                        ),
                    ],
                ),
            },
        )
        agent = config.agents["test"]
        assert agent.approval_tools == {"dangerous_fn", "write_data"}

    def test_no_approval_tools(self):
        config = AgentsConfig(
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        assert config.agents["test"].approval_tools == set()

    def test_yaml_round_trip(self):
        raw = {
            "tools": {
                "send_alert": {
                    "handler": "mymod.alert",
                    "requires_approval": True,
                },
            },
            "agents": {
                "ops": {
                    "description": "Ops agent",
                    "prompt": "You are ops",
                    "tools": ["send_alert"],
                    "mcp_servers": [
                        {
                            "name": "infra",
                            "url": "http://localhost:9090",
                            "tools": [
                                {"name": "restart_service", "requires_approval": True},
                                {"name": "get_status"},
                            ],
                        },
                    ],
                },
            },
        }
        config = AgentsConfig(**raw)
        assert config.agents["ops"].approval_tools == {"send_alert", "restart_service"}


# ── Tool wrapper tests: requires_approval propagation ───────


class TestToolWrapperApproval:
    """Tool wrappers carry the requires_approval flag."""

    def test_mcp_wrapper_default(self):
        from orchid_ai.agents.tools import MCPToolWrapper

        tool = MCPToolWrapper(
            name="test",
            description="test",
            mcp_client=MagicMock(),
            auth=MagicMock(),
        )
        assert tool.requires_approval is False

    def test_mcp_wrapper_with_approval(self):
        from orchid_ai.agents.tools import MCPToolWrapper

        tool = MCPToolWrapper(
            name="test",
            description="test",
            mcp_client=MagicMock(),
            auth=MagicMock(),
            requires_approval=True,
        )
        assert tool.requires_approval is True

    def test_builtin_wrapper_with_approval(self):
        from orchid_ai.agents.tools import BuiltinToolWrapper

        tool = BuiltinToolWrapper(
            name="test",
            description="test",
            auth=MagicMock(),
            requires_approval=True,
        )
        assert tool.requires_approval is True


class TestBuildLangchainToolsApproval:
    """build_langchain_tools() propagates approval_tools to wrappers."""

    def test_approval_tools_set_on_wrappers(self):
        from orchid_ai.agents.tools import build_langchain_tools
        from orchid_ai.core.state import AuthContext

        mock_client = MagicMock()
        auth = AuthContext(access_token="test")

        tools = build_langchain_tools(
            builtin_names={"safe_fn", "dangerous_fn"},
            builtin_tool_defs=[
                {"type": "function", "function": {"name": "safe_fn", "description": "Safe", "parameters": {}}},
                {"type": "function", "function": {"name": "dangerous_fn", "description": "Danger", "parameters": {}}},
            ],
            mcp_tool_defs=[
                {"type": "function", "function": {"name": "mcp_read", "description": "Read", "parameters": {}}},
                {"type": "function", "function": {"name": "mcp_write", "description": "Write", "parameters": {}}},
            ],
            mcp_tool_client_map={
                "mcp_read": (mock_client, MagicMock()),
                "mcp_write": (mock_client, MagicMock()),
            },
            auth=auth,
            approval_tools={"dangerous_fn", "mcp_write"},
        )

        tool_map = {t.name: t for t in tools}
        assert tool_map["safe_fn"].requires_approval is False
        assert tool_map["dangerous_fn"].requires_approval is True
        assert tool_map["mcp_read"].requires_approval is False
        assert tool_map["mcp_write"].requires_approval is True
