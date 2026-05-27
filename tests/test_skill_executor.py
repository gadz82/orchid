"""Tests for SkillExecutor — built-in tool step signature introspection."""

from __future__ import annotations

import pytest

from orchid_ai.agents.skill_executor import SkillExecutor
from orchid_ai.config import tool_registry as treg
from orchid_ai.core.state import OrchidAuthContext


@pytest.fixture(autouse=True)
def _clean_registry():
    treg.clear()
    yield
    treg.clear()


def _make_auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="tok", tenant_key="t", user_id="u")


def _make_executor() -> SkillExecutor:
    from orchid_ai.core.agent import OrchidAgent

    return SkillExecutor(
        agent_name="test",
        mcp_dispatcher=None,
        builtin_tool_caller=OrchidAgent.call_builtin_tool.__get__(type("Stub", (), {})),  # won't be used directly
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
    async def test_tool_with_kwargs_does_not_leak_framework_params(self):
        """Tools with **kwargs do NOT receive framework params unless explicitly declared."""
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
        # Framework params are stripped — only business params reach **kwargs.
        assert "query" not in received_keys
        assert "auth_context" not in received_keys
        assert "context" not in received_keys
        assert "content_sources" not in received_keys
        assert "extra" in received_keys

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


class TestNoAutoMapping:
    """Verify that no implicit query-to-parameter mapping occurs.

    Tools that need specific parameters must receive them via
    step_arguments in YAML.  For workflows requiring LLM-driven
    parameter extraction, use orchestrator-level skills with
    agent steps instead of deterministic tool chaining.
    """

    @pytest.mark.asyncio
    async def test_explicit_step_arguments_are_forwarded(self):
        """step_arguments values reach the handler correctly."""
        received = {}

        async def search_courses(*, auth_context, search_text: str, **_kw):
            received["search_text"] = search_text
            return {"courses": []}

        treg.register_tool("search_courses", search_courses, "Search courses")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        await executor._run_builtin_step(
            "search_courses",
            query="user said something",
            auth=_make_auth(),
            step_arguments={"search_text": "Course C1"},
            previous_results={},
        )
        assert received["search_text"] == "Course C1"

    @pytest.mark.asyncio
    async def test_missing_required_param_raises(self):
        """Without step_arguments, required params cause a TypeError."""

        async def tool_with_required(*, auth_context, course_id: int, **_kw):
            return {"course_id": course_id}

        treg.register_tool("strict_tool", tool_with_required, "Strict tool")

        async def caller(name: str, **kwargs):
            return await treg.call_tool(name, **kwargs)

        executor = SkillExecutor(
            agent_name="test",
            mcp_dispatcher=None,
            builtin_tool_caller=caller,
        )

        with pytest.raises(TypeError, match="course_id"):
            await executor._run_builtin_step(
                "strict_tool",
                query="some text",
                auth=_make_auth(),
                step_arguments={},
                previous_results={},
            )
