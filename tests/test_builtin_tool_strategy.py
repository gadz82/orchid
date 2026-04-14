"""Tests for built-in tool LLM-decides execution in GenericAgent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import AgentConfig, LLMConfig, RAGConfig
from orchid_ai.config.tool_registry import BuiltinToolEntry, ToolParameter
from orchid_ai.core.state import AuthContext


def _make_auth() -> AuthContext:
    return AuthContext(access_token="tok", tenant_key="t1", user_id="u1")


def _make_agent(tools: list[str], llm_service: Any = None) -> GenericAgent:
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
    if llm_service is None:
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(return_value="[]")
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


class TestBuiltinToolsLLMDecides:
    """The LLM decides which built-in tools to call and with what arguments."""

    @pytest.mark.asyncio
    async def test_calls_only_decided_tools(self):
        """LLM decides to call tool_a with specific args, skips tool_b."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(
            return_value=json.dumps(
                [
                    {"tool": "tool_a", "arguments": {"search_text": "Course C1"}},
                ]
            )
        )
        agent = _make_agent(["tool_a", "tool_b"], llm_service=llm_service)

        entry_a = _mock_tool_entry(
            "tool_a",
            "Search courses",
            {
                "search_text": ToolParameter(name="search_text", description="Keyword"),
            },
        )
        entry_b = _mock_tool_entry(
            "tool_b",
            "Get enrollments",
            {
                "course_id": ToolParameter(name="course_id", type="int", description="Course ID"),
            },
        )

        with (
            patch(
                "orchid_ai.config.tool_registry.get_tool",
                side_effect=lambda n: {"tool_a": entry_a, "tool_b": entry_b}[n],
            ),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            mock_call.return_value = {"courses": []}
            results = await agent._run_builtin_tools(
                "Find course C1",
                {},
                auth_context=_make_auth(),
            )

        assert mock_call.call_count == 1
        call_args = mock_call.call_args
        assert call_args[0][0] == "tool_a"
        assert call_args[1]["search_text"] == "Course C1"
        assert "builtin_tool_a" in results
        assert "builtin_tool_b" not in results

    @pytest.mark.asyncio
    async def test_passes_auth_context(self):
        """auth_context is always injected into the tool kwargs."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(
            return_value=json.dumps(
                [
                    {"tool": "tool_a", "arguments": {"search_text": "test"}},
                ]
            )
        )
        agent = _make_agent(["tool_a"], llm_service=llm_service)

        entry_a = _mock_tool_entry(
            "tool_a",
            "Search",
            {
                "search_text": ToolParameter(name="search_text", description="Keyword"),
            },
        )
        auth = _make_auth()

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            mock_call.return_value = "ok"
            await agent._run_builtin_tools("q", {}, auth_context=auth)

        assert mock_call.call_args[1]["auth_context"] is auth

    @pytest.mark.asyncio
    async def test_empty_array_calls_nothing(self):
        """When LLM returns [], no tools are called."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(return_value="[]")
        agent = _make_agent(["tool_a"], llm_service=llm_service)

        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            results = await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        mock_call.assert_not_called()
        assert results == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_bad_json(self):
        """When LLM returns invalid JSON, returns empty results."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(return_value="not json at all")
        agent = _make_agent(["tool_a"], llm_service=llm_service)

        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            results = await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        mock_call.assert_not_called()
        assert results == {}

    @pytest.mark.asyncio
    async def test_returns_empty_without_llm_service(self):
        """Without LLMProvider, returns empty results."""
        agent = _make_agent(["tool_a"])
        agent._llm_service = None

        with patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call:
            results = await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        mock_call.assert_not_called()
        assert results == {}

    @pytest.mark.asyncio
    async def test_ignores_unknown_tool(self):
        """LLM decides a tool not in the agent's list — it's skipped."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(
            return_value=json.dumps(
                [
                    {"tool": "nonexistent_tool", "arguments": {}},
                ]
            )
        )
        agent = _make_agent(["tool_a"], llm_service=llm_service)

        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            results = await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        mock_call.assert_not_called()
        assert results == {}

    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        """LLM can decide to call multiple tools."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(
            return_value=json.dumps(
                [
                    {"tool": "search", "arguments": {"search_text": "C1"}},
                    {"tool": "enroll", "arguments": {"course_id": 42}},
                ]
            )
        )
        agent = _make_agent(["search", "enroll"], llm_service=llm_service)

        entries = {
            "search": _mock_tool_entry(
                "search",
                "Search courses",
                {
                    "search_text": ToolParameter(name="search_text", description="Keyword"),
                },
            ),
            "enroll": _mock_tool_entry(
                "enroll",
                "Get enrollments",
                {
                    "course_id": ToolParameter(name="course_id", type="int", description="ID"),
                },
            ),
        }

        with (
            patch("orchid_ai.config.tool_registry.get_tool", side_effect=lambda n: entries[n]),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            mock_call.return_value = "ok"
            results = await agent._run_builtin_tools(
                "Find incomplete users for course C1",
                {},
                auth_context=_make_auth(),
            )

        assert mock_call.call_count == 2
        assert "builtin_search" in results
        assert "builtin_enroll" in results

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_error(self):
        """When the LLM call raises, returns empty results."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = _make_agent(["tool_a"], llm_service=llm_service)

        entry_a = _mock_tool_entry("tool_a", "Search", {})

        with (
            patch("orchid_ai.config.tool_registry.get_tool", return_value=entry_a),
            patch.object(agent, "call_builtin_tool", new_callable=AsyncMock) as mock_call,
        ):
            results = await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        mock_call.assert_not_called()
        assert results == {}

    @pytest.mark.asyncio
    async def test_prompt_includes_parameter_metadata(self):
        """The decision prompt sent to the LLM includes tool parameter details."""
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(return_value="[]")
        agent = _make_agent(["my_tool"], llm_service=llm_service)

        entry = _mock_tool_entry(
            "my_tool",
            "Does things",
            {
                "name": ToolParameter(name="name", type="string", description="The name to search"),
                "limit": ToolParameter(name="limit", type="int", description="Max results", required=False, default=10),
            },
        )

        with patch("orchid_ai.config.tool_registry.get_tool", return_value=entry):
            await agent._run_builtin_tools("q", {}, auth_context=_make_auth())

        prompt_sent = llm_service.complete.call_args[0][1][0]["content"]
        assert "my_tool: Does things" in prompt_sent
        assert "name (string, required): The name to search" in prompt_sent
        assert "limit (int, optional, default=10): Max results" in prompt_sent
