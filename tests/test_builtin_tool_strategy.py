"""Tests for the unified agentic tool-calling loop in GenericAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
from orchid_ai.config.schema import AgentConfig, LLMConfig, RAGConfig
from orchid_ai.config.tool_registry import BuiltinToolEntry, ToolParameter
from orchid_ai.core.state import AuthContext


def _make_auth() -> AuthContext:
    return AuthContext(access_token="tok", tenant_key="t1", user_id="u1")


def _make_agent(tools: list[str]) -> GenericAgent:
    config = AgentConfig(
        name="test_agent",
        description="test",
        prompt="test prompt",
        tools=tools,
        rag=RAGConfig(enabled=False),
        llm=LLMConfig(),
    )
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])
    llm_service = MagicMock()
    llm_service.complete = AsyncMock(return_value="summary")
    return GenericAgent(
        config=config,
        llm="test-model",
        reader=reader,
        mcp_clients=[],
        llm_service=llm_service,
    )


def _mock_tool_entry(name: str, description: str, params: dict[str, ToolParameter]) -> BuiltinToolEntry:
    return BuiltinToolEntry(
        name=name,
        handler=AsyncMock(),
        description=description,
        parameters=params,
    )


def _make_resp(content: str | None = None, tool_calls: list | None = None):
    """Build a mock litellm acompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {"role": "assistant", "content": content}
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_tc(name: str, args: dict, call_id: str = "tc_1"):
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    tc.id = call_id
    return tc


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

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value={"found": "data"}),
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("tool_a", {"search_text": "test"})]),
                _make_resp(content="Here are the results."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test query",
                _make_auth(),
                None,
                [],
            )

        assert final_text == "Here are the results."
        assert "tool_a" in results

    @pytest.mark.asyncio
    async def test_multi_turn_dependent_tools(self):
        """LLM calls tool A, sees results, then calls tool B with derived args."""
        agent = _make_agent(["search", "enroll"])
        entries = {
            "search": _mock_tool_entry(
                "search", "Search items", {"search_text": ToolParameter(name="search_text", description="Keyword")}
            ),
            "enroll": _mock_tool_entry(
                "enroll", "Get details", {"item_id": ToolParameter(name="item_id", type="int", description="ID")}
            ),
        }

        call_order: list[str] = []

        async def _track_call(name, **kwargs):
            call_order.append(name)
            if name == "search":
                return {"items": [{"id": 42}]}
            return {"details": [{"user": "alice"}]}

        with (
            patch("orchid_ai.config.tool_registry.get_tool", side_effect=lambda n: entries[n]),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", side_effect=_track_call),
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("search", {"search_text": "C1"}, "tc1")]),
                _make_resp(tool_calls=[_make_tc("enroll", {"item_id": 42}, "tc2")]),
                _make_resp(content="Found details."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "Find details for C1",
                _make_auth(),
                None,
                [],
            )

        assert call_order == ["search", "enroll"]
        assert "search" in results
        assert "enroll" in results
        assert final_text == "Found details."

    @pytest.mark.asyncio
    async def test_duplicate_call_detection(self):
        """Duplicate tool call returns cached result with warning, not re-execution."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry("tool_a", "Search", {"q": ToolParameter(name="q", description="Query")})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value={"data": "x"}) as mock_call,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("tool_a", {"q": "test"}, "tc1")]),
                _make_resp(tool_calls=[_make_tc("tool_a", {"q": "test"}, "tc2")]),
                _make_resp(content="Done."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        # Built-in tool only called ONCE (second is a cached duplicate)
        assert mock_call.call_count == 1
        assert "already called this tool" in results["tool_a"]

    @pytest.mark.asyncio
    async def test_consecutive_dupes_force_text_response(self):
        """After _MAX_CONSECUTIVE_DUPES, tools are stripped to force text response."""
        agent = _make_agent(["tool_a"])
        agent._MAX_CONSECUTIVE_DUPES = 1

        entry_a = _mock_tool_entry("tool_a", "Search", {"q": ToolParameter(name="q", description="Query")})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value="ok"),
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("tool_a", {"q": "x"}, "tc1")]),
                _make_resp(tool_calls=[_make_tc("tool_a", {"q": "x"}, "tc2")]),
                _make_resp(content="Forced summary."),
            ]
            final_text, _ = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert final_text == "Forced summary."
        # Third call should NOT have "tools" in kwargs (stripped)
        third_call_kwargs = mock_ac.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    @pytest.mark.asyncio
    async def test_error_503_returns_message(self):
        """503 error from LLM returns user-friendly message."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("503 Service Unavailable")),
        ):
            final_text, _ = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert "high demand" in final_text

    @pytest.mark.asyncio
    async def test_error_rate_limit_returns_message(self):
        """Rate limit error from LLM returns user-friendly message."""
        agent = _make_agent(["tool_a"])
        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("Rate limit exceeded")),
        ):
            final_text, _ = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert "Rate limit" in final_text

    @pytest.mark.asyncio
    async def test_no_tools_returns_none(self):
        """When no tools are available, returns (None, {})."""
        agent = _make_agent([])

        with patch.object(
            agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
        ):
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert final_text is None
        assert results == {}

    @pytest.mark.asyncio
    async def test_max_rounds_respected(self):
        """Loop stops after _MAX_TOOL_ROUNDS even if LLM keeps calling tools."""
        agent = _make_agent(["tool_a"])
        agent._MAX_TOOL_ROUNDS = 3

        entry_a = _mock_tool_entry("tool_a", "Search", {"q": ToolParameter(name="q", description="Query")})

        responses = [_make_resp(tool_calls=[_make_tc("tool_a", {"q": f"x{i}"}, f"tc{i}")]) for i in range(3)]

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(
                agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=_EMPTY_CAPS
            ),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value="ok"),
        ):
            mock_ac.side_effect = responses
            final_text, _ = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert final_text is None
        assert mock_ac.call_count == 3

    @pytest.mark.asyncio
    async def test_mcp_tool_called_via_client(self):
        """MCP tools are routed through the correct MCP client."""
        agent = _make_agent([])

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = '{"items": [1, 2]}'
        mock_result.is_error = False
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        caps = MCPCapabilities(
            raw_tools=[
                {"name": "mcp_search", "description": "Search via MCP", "schema": {"type": "object", "properties": {}}}
            ],
            tool_client_map={"mcp_search": (mock_client, MagicMock())},
        )

        with (
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("mcp_search", {"q": "test"}, "tc1")]),
                _make_resp(content="Found results."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        mock_client.call_tool.assert_called_once_with("mcp_search", {"q": "test"}, _make_auth())
        assert "mcp_search" in results
        assert final_text == "Found results."

    @pytest.mark.asyncio
    async def test_builtin_wins_over_mcp_overlap(self):
        """When a built-in tool and MCP tool share a name, built-in wins."""
        agent = _make_agent(["shared_tool"])
        entry = _mock_tool_entry("shared_tool", "Built-in version", {"q": ToolParameter(name="q", description="Query")})

        mock_client = MagicMock()
        caps = MCPCapabilities(
            raw_tools=[{"name": "shared_tool", "description": "MCP version", "schema": {"type": "object"}}],
            tool_client_map={"shared_tool": (mock_client, MagicMock())},
        )

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry),
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(
                agent, "call_builtin_tool", new_callable=AsyncMock, return_value="builtin_result"
            ) as mock_builtin,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("shared_tool", {"q": "x"}, "tc1")]),
                _make_resp(content="Done."),
            ]
            _, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        mock_builtin.assert_called_once()
        mock_client.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_system_prompt_includes_rag_and_resources(self):
        """System prompt includes RAG data, rendered prompts, and resources."""
        agent = _make_agent([])

        # Need at least one MCP tool so the loop actually runs and calls litellm
        mock_client = MagicMock()
        caps = MCPCapabilities(
            raw_tools=[{"name": "some_tool", "description": "A tool", "schema": {"type": "object"}}],
            tool_client_map={"some_tool": (mock_client, MagicMock())},
            rendered_prompts=[{"name": "guide", "text": "Use tools wisely."}],
            resource_contents={"config": "max_retries: 3"},
            skipped_prompts=[{"name": "complex", "description": "Needs args", "required_args": ["entity_id"]}],
        )
        rag_data = [{"content": "Background info", "score": 0.9}]

        with (
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
        ):
            mock_ac.return_value = _make_resp(content="Summary.")
            await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                rag_data,
            )

        call_kwargs = mock_ac.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Use tools wisely." in system_msg
        assert "max_retries: 3" in system_msg
        assert "Background info" in system_msg
        assert "complex" in system_msg
        assert "entity_id" in system_msg

    @pytest.mark.asyncio
    async def test_mixed_builtin_and_mcp_in_same_round(self):
        """LLM calls both a built-in tool and an MCP tool in the same round."""
        agent = _make_agent(["builtin_tool"])
        entry = _mock_tool_entry(
            "builtin_tool",
            "Local tool",
            {"q": ToolParameter(name="q", description="Query")},
        )

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = '{"remote": true}'
        mock_result.is_error = False
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        caps = MCPCapabilities(
            raw_tools=[
                {"name": "mcp_tool", "description": "Remote tool", "schema": {"type": "object", "properties": {}}}
            ],
            tool_client_map={"mcp_tool": (mock_client, MagicMock())},
        )

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry),
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock, return_value={"local": True}),
        ):
            # LLM emits two tool_calls in a single response
            mock_ac.side_effect = [
                _make_resp(
                    tool_calls=[
                        _make_tc("builtin_tool", {"q": "x"}, "tc1"),
                        _make_tc("mcp_tool", {"q": "y"}, "tc2"),
                    ]
                ),
                _make_resp(content="Combined results."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert final_text == "Combined results."
        assert "builtin_tool" in results
        assert "mcp_tool" in results
        mock_client.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tool_error_is_reported(self):
        """When an MCP tool raises an exception, the error is captured in results."""
        agent = _make_agent([])

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=ConnectionError("server down"))

        caps = MCPCapabilities(
            raw_tools=[
                {"name": "flaky_tool", "description": "Unreliable", "schema": {"type": "object", "properties": {}}}
            ],
            tool_client_map={"flaky_tool": (mock_client, MagicMock())},
        )

        with (
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("flaky_tool", {}, "tc1")]),
                _make_resp(content="Tool failed, here is a summary."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert "[Tool error]" in results["flaky_tool"]
        assert "server down" in results["flaky_tool"]
        assert final_text == "Tool failed, here is a summary."

    @pytest.mark.asyncio
    async def test_mcp_tool_is_error_flag(self):
        """When an MCP tool returns is_error=True, the result is prefixed with [Tool error]."""
        agent = _make_agent([])

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Invalid parameters"
        mock_result.is_error = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        caps = MCPCapabilities(
            raw_tools=[
                {"name": "strict_tool", "description": "Validates", "schema": {"type": "object", "properties": {}}}
            ],
            tool_client_map={"strict_tool": (mock_client, MagicMock())},
        )

        with (
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("strict_tool", {"x": 1}, "tc1")]),
                _make_resp(content="Handled the error."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert results["strict_tool"] == "[Tool error] Invalid parameters"
        assert final_text == "Handled the error."

    @pytest.mark.asyncio
    async def test_unknown_tool_name_reported(self):
        """When the LLM hallucinates a tool name, an error is recorded."""
        agent = _make_agent([])

        # Caps with one real tool, but LLM will call a different name
        mock_client = MagicMock()
        caps = MCPCapabilities(
            raw_tools=[{"name": "real_tool", "description": "Exists", "schema": {"type": "object", "properties": {}}}],
            tool_client_map={"real_tool": (mock_client, MagicMock())},
        )

        with (
            patch.object(agent._mcp_dispatcher, "render_capabilities", new_callable=AsyncMock, return_value=caps),
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac,
        ):
            mock_ac.side_effect = [
                _make_resp(tool_calls=[_make_tc("hallucinated_tool", {"q": "x"}, "tc1")]),
                _make_resp(content="Recovered."),
            ]
            final_text, results = await agent._agentic_tool_loop(
                "test",
                _make_auth(),
                None,
                [],
            )

        assert "[Error] Unknown tool 'hallucinated_tool'" in results["hallucinated_tool"]
        assert final_text == "Recovered."
