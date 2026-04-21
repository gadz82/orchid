"""Tests for the unified agentic tool-calling loop in GenericAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
from orchid_ai.config.schema import OrchidAgentConfig, OrchidLLMConfig, OrchidRAGConfig
from orchid_ai.config.tool_registry import BuiltinToolEntry, ToolParameter
from orchid_ai.core.state import OrchidAuthContext


def _make_auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="tok", tenant_key="t1", user_id="u1")


def _make_ai_message(content: str | None = None, tool_calls: list | None = None):
    """Build a mock LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content or ""
    msg.tool_calls = tool_calls or []
    return msg


def _make_agent(tools: list[str]) -> GenericAgent:
    config = OrchidAgentConfig(
        name="test_agent",
        description="test",
        prompt="test prompt",
        tools=tools,
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(),
    )
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=_make_ai_message(content="summary"))
    chat_model.bind_tools = MagicMock(return_value=chat_model)  # bind_tools returns self
    return GenericAgent(
        config=config,
        llm="test-model",
        reader=reader,
        mcp_clients=[],
        chat_model=chat_model,
    )


def _mock_tool_entry(name: str, description: str, params: dict[str, ToolParameter]) -> BuiltinToolEntry:
    return BuiltinToolEntry(
        name=name,
        handler=AsyncMock(),
        description=description,
        parameters=params,
    )


_EMPTY_CAPS = MCPCapabilities()


class TestAgenticToolLoop:
    """Tests for GenericAgent._agentic_tool_loop with native tool_calls."""

    @pytest.mark.asyncio
    async def test_calls_builtin_tool_and_returns_text(self):
        """LLM calls a built-in tool, then produces a text response."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry(
            "tool_a",
            "Search",
            {"search_text": ToolParameter(name="search_text", description="Keyword")},
        )

        # Mock: first call returns tool_call, second returns text
        tool_call_msg = _make_ai_message(tool_calls=[{"name": "tool_a", "args": {"search_text": "test"}, "id": "tc_1"}])
        final_msg = _make_ai_message(content="Here are the results.")

        agent._chat_model.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])
        agent._chat_model.bind_tools = MagicMock(return_value=agent._chat_model)

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value={"found": "data"}),
        ):
            final_text, results = await agent._agentic_tool_loop(
                "test query",
                _make_auth(),
                None,
                [],
            )

        assert final_text == "Here are the results."
        assert "tool_a" in results

    @pytest.mark.asyncio
    async def test_no_tools_returns_none(self):
        """When no tools are available, returns (None, {})."""
        agent = _make_agent([])  # no built-in tools

        with patch.object(
            agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
        ):
            final_text, results = await agent._agentic_tool_loop(
                "test query",
                _make_auth(),
                None,
                [],
            )

        assert final_text is None
        assert results == {}

    @pytest.mark.asyncio
    async def test_immediate_text_response(self):
        """LLM responds with text immediately (no tool calls)."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry(
            "tool_a",
            "Search",
            {"search_text": ToolParameter(name="search_text", description="Keyword")},
        )

        text_msg = _make_ai_message(content="I don't need tools for this.")
        agent._chat_model.ainvoke = AsyncMock(return_value=text_msg)
        agent._chat_model.bind_tools = MagicMock(return_value=agent._chat_model)

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
        ):
            final_text, results = await agent._agentic_tool_loop(
                "test query",
                _make_auth(),
                None,
                [],
            )

        assert final_text == "I don't need tools for this."
        assert results == {}

    @pytest.mark.asyncio
    async def test_llm_error_returns_error_message(self):
        """LLM API error returns a user-friendly error message."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry(
            "tool_a",
            "Search",
            {"search_text": ToolParameter(name="search_text", description="Keyword")},
        )

        agent._chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("connection failed"))
        agent._chat_model.bind_tools = MagicMock(return_value=agent._chat_model)

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
        ):
            final_text, results = await agent._agentic_tool_loop(
                "test query",
                _make_auth(),
                None,
                [],
            )

        assert final_text is not None
        assert "connection failed" in final_text
