"""Tests for SkillExecutor — built-in tool step signature introspection."""

from __future__ import annotations

import pytest

from orchid_ai.agents.skill_executor import SkillExecutor
from orchid_ai.config import tool_registry as treg
from orchid_ai.core.state import AuthContext


@pytest.fixture(autouse=True)
def _clean_registry():
    treg.clear()
    yield
    treg.clear()


def _make_auth() -> AuthContext:
    return AuthContext(access_token="tok", tenant_key="t", user_id="u")


def _make_executor() -> SkillExecutor:
    from orchid_ai.core.agent import BaseAgent

    return SkillExecutor(
        agent_name="test",
        mcp_dispatcher=None,
        builtin_tool_caller=BaseAgent.call_builtin_tool.__get__(type("Stub", (), {})),  # won't be used directly
    )


class TestBuiltinStepSignatureFiltering:
    """Verify that _run_builtin_step only passes kwargs the handler accepts."""

    @pytest.mark.asyncio
    async def test_tool_without_query_param(self):
        """Tools like calculate_completion_rate(enrolled, completed) must not receive 'query'."""

        def calc(enrolled: int, completed: int) -> float:
            return round(completed / enrolled * 100, 1) if enrolled > 0 else 0.0

        treg.register_tool("calc", calc, "Calculate rate")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        result = await executor._run_builtin_step(
            "calc",
            query="what is the rate?",
            auth=_make_auth(),
            step_arguments={"enrolled": 100, "completed": 75},
            previous_results={},
        )
        assert result == 75.0

    @pytest.mark.asyncio
    async def test_tool_with_auth_context(self):
        """Tools requiring auth_context receive it from the skill executor."""
        received = {}

        async def api_tool(*, auth_context, search_text: str = "", **_kwargs):
            received["auth"] = auth_context
            received["search_text"] = search_text
            return {"found": True}

        treg.register_tool("api_tool", api_tool, "API tool")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        result = await executor._run_builtin_step(
            "api_tool",
            query="find something",
            auth=_make_auth(),
            step_arguments={"search_text": "test query"},
            previous_results={},
        )
        assert result == {"found": True}
        assert received["auth"].access_token == "tok"
        assert received["search_text"] == "test query"

    @pytest.mark.asyncio
    async def test_tool_with_kwargs_receives_everything(self):
        """Tools with **kwargs get all available params."""
        received_keys = set()

        def flexible_tool(**kwargs):
            received_keys.update(kwargs.keys())
            return "ok"

        treg.register_tool("flex", flexible_tool, "Flexible")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        await executor._run_builtin_step(
            "flex",
            query="hello",
            auth=_make_auth(),
            step_arguments={"extra": "val"},
            previous_results={"prev": "data"},
        )
        assert "query" in received_keys
        assert "auth_context" in received_keys
        assert "extra" in received_keys
        assert "context" in received_keys

    @pytest.mark.asyncio
    async def test_previous_results_as_context(self):
        """Previous step results are passed as 'context' when handler accepts it."""
        received = {}

        def tool_with_context(query: str, context: dict = None):
            received["query"] = query
            received["context"] = context
            return "done"

        treg.register_tool("ctx_tool", tool_with_context, "Context tool")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        result = await executor._run_builtin_step(
            "ctx_tool",
            query="test",
            auth=_make_auth(),
            step_arguments={},
            previous_results={"step1": "result1"},
        )
        assert result == "done"
        assert received["context"] == {"step1": "result1"}

    @pytest.mark.asyncio
    async def test_step_arguments_override_defaults(self):
        """Explicit step arguments override framework-provided defaults."""

        def search(query: str = "", limit: int = 10) -> dict:
            return {"query": query, "limit": limit}

        treg.register_tool("search", search, "Search")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        result = await executor._run_builtin_step(
            "search",
            query="user question",
            auth=_make_auth(),
            step_arguments={"limit": 5},
            previous_results={},
        )
        assert result["query"] == "user question"
        assert result["limit"] == 5
