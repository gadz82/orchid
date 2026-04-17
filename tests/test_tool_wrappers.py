"""Tests for LangChain BaseTool wrappers (#11)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.tools import (
    BuiltinToolWrapper,
    MCPToolWrapper,
    build_langchain_tools,
)
from orchid_ai.core.state import AuthContext


def _make_auth() -> AuthContext:
    return AuthContext(access_token="test-token")


# ── MCPToolWrapper tests ────────────────────────────────────


class TestMCPToolWrapper:
    """MCPToolWrapper wraps MCP client calls."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = '{"data": "value"}'
        mock_result.is_error = False
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool = MCPToolWrapper(
            name="list_items",
            description="List items",
            mcp_client=mock_client,
            auth=_make_auth(),
            agent_name="test",
        )

        result = await tool.ainvoke({"filter": "active"})
        assert result == '{"data": "value"}'
        mock_client.call_tool.assert_called_once_with("list_items", {"filter": "active"}, tool.auth)

    @pytest.mark.asyncio
    async def test_error_result(self):
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Invalid argument"
        mock_result.is_error = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool = MCPToolWrapper(
            name="bad_tool",
            description="Fails",
            mcp_client=mock_client,
            auth=_make_auth(),
        )

        result = await tool.ainvoke({})
        assert result.startswith("[Tool error]")

    @pytest.mark.asyncio
    async def test_exception_handled(self):
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=ConnectionError("timeout"))

        tool = MCPToolWrapper(
            name="flaky_tool",
            description="Flaky",
            mcp_client=mock_client,
            auth=_make_auth(),
        )

        result = await tool.ainvoke({})
        assert "[Tool error]" in result


# ── BuiltinToolWrapper tests ────────────────────────────────


class TestBuiltinToolWrapper:
    """BuiltinToolWrapper wraps registered built-in tools."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        tool = BuiltinToolWrapper(
            name="echo_tool",
            description="Echo",
            auth=_make_auth(),
            agent_name="test",
        )

        # Register a mock tool
        from orchid_ai.config.tool_registry import register_tool

        async def echo_handler(**kwargs):
            return {"echo": kwargs.get("message", "")}

        register_tool("echo_tool", echo_handler, "Echo tool")

        result = await tool.ainvoke({"message": "hello"})
        parsed = json.loads(result)
        assert parsed["echo"] == "hello"

    @pytest.mark.asyncio
    async def test_exception_handled(self):
        tool = BuiltinToolWrapper(
            name="nonexistent_tool_xyz",
            description="Missing",
            auth=_make_auth(),
        )

        result = await tool.ainvoke({})
        assert "[Tool error]" in result


# ── build_langchain_tools tests ─────────────────────────────


class TestBuildLangchainTools:
    """build_langchain_tools() creates a unified tool list."""

    def test_builds_mixed_tools(self):
        mock_client = MagicMock()
        auth = _make_auth()

        tools = build_langchain_tools(
            builtin_names={"search"},
            builtin_tool_defs=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search things",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            mcp_tool_defs=[
                {
                    "type": "function",
                    "function": {
                        "name": "create_item",
                        "description": "Create an item",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            mcp_tool_client_map={"create_item": (mock_client, MagicMock())},
            auth=auth,
            agent_name="test",
        )

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search" in names
        assert "create_item" in names

        # Verify types
        builtin = next(t for t in tools if t.name == "search")
        mcp = next(t for t in tools if t.name == "create_item")
        assert isinstance(builtin, BuiltinToolWrapper)
        assert isinstance(mcp, MCPToolWrapper)

    def test_empty_tools(self):
        tools = build_langchain_tools(
            builtin_names=set(),
            builtin_tool_defs=[],
            mcp_tool_defs=[],
            mcp_tool_client_map={},
            auth=_make_auth(),
        )
        assert tools == []

    def test_skips_mcp_tool_without_client(self):
        tools = build_langchain_tools(
            builtin_names=set(),
            builtin_tool_defs=[],
            mcp_tool_defs=[
                {
                    "type": "function",
                    "function": {
                        "name": "orphan_tool",
                        "description": "No client",
                        "parameters": {},
                    },
                },
            ],
            mcp_tool_client_map={},  # no mapping for orphan_tool
            auth=_make_auth(),
        )
        assert len(tools) == 0
