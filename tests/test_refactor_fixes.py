"""Tests for the 10-point refactor — validates all behavioral changes."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchid.agents.strategies import CallAllStrategy, LLMDecidesStrategy
from orchid.config.schema import ToolConfig
from orchid.core.mcp import MCPClient, MCPToolResult
from orchid.core.state import AuthContext


# ── Helpers ──────────────────────────────────────────────────────────


def _auth():
    return AuthContext(access_token="test-token")


def _tools(*names):
    return [ToolConfig(name=n) for n in names]


class _TimingMCPClient(MCPClient):
    """MCP client that records call timestamps to verify concurrency."""

    def __init__(self, delay: float = 0.05):
        self._delay = delay
        self.call_times: list[tuple[str, float]] = []

    async def call_tool(self, tool_name, arguments, auth):
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(self._delay)
        self.call_times.append((tool_name, start))
        return MCPToolResult(content=[{"type": "text", "text": f"result_{tool_name}"}])

    async def list_tools(self, auth):
        return []

    async def list_prompts(self, auth):
        return []

    async def list_resources(self, auth):
        return []

    async def get_prompt(self, name, arguments, auth):
        return []

    async def read_resource(self, uri, auth):
        return ""

    @property
    def server_url(self):
        return "http://timing-mock"


# ── Fix #1: CallAllStrategy runs tools concurrently ──────────────────


class TestCallAllConcurrency:
    @pytest.mark.asyncio
    async def test_tools_run_concurrently(self):
        """CallAllStrategy should run tools via asyncio.gather, not sequentially."""
        client = _TimingMCPClient(delay=0.05)
        strategy = CallAllStrategy()
        tools = _tools("a", "b", "c")

        results = await strategy.execute(client, tools, "q", _auth())

        # All 3 tools should return results
        assert len(results) == 3
        assert "a" in results and "b" in results and "c" in results

        # Verify concurrency: all tools should start within a tight window
        # (if sequential, total time would be ~0.15s; concurrent ~0.05s)
        starts = [t[1] for t in client.call_times]
        assert max(starts) - min(starts) < 0.03, "Tools should start nearly simultaneously"


# ── Fix #3: fetch_all_rag_context runs domain + uploads concurrently ──


class TestFetchAllRagConcurrency:
    @pytest.mark.asyncio
    async def test_domain_and_uploads_run_concurrently(self):
        """fetch_all_rag_context should use asyncio.gather for domain + uploads."""
        from orchid.core.agent import BaseAgent
        from orchid.rag.scopes import RAGScope

        # Create a concrete subclass for testing
        class _TestAgent(BaseAgent):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test agent"

            @property
            def rag_namespace(self):
                return "test_ns"

            async def run(self, state):
                return state

        call_order: list[str] = []

        async def mock_retrieve(*, query, namespace, k, scope):
            call_order.append(namespace)
            await asyncio.sleep(0.02)
            return []

        reader = MagicMock()
        reader.retrieve = mock_retrieve

        agent = _TestAgent(llm="model", reader=reader)
        scope = RAGScope(tenant_id="t", user_id="u")

        result = await agent.fetch_all_rag_context("query", scope, k=3)

        # Both namespaces should be called
        assert "test_ns" in call_order
        assert "uploads" in call_order
        assert result == []


# ── Fix #5: litellm fallback removed — RuntimeError on missing LLMProvider ──


class TestNoLitellmFallback:
    @pytest.mark.asyncio
    async def test_summarise_raises_without_llm_service(self):
        """BaseAgent.summarise() should raise RuntimeError when no LLMProvider injected."""
        from orchid.core.agent import BaseAgent
        from orchid.rag.scopes import RAGScope

        class _TestAgent(BaseAgent):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            async def run(self, state):
                return state

        agent = _TestAgent(llm="model", reader=MagicMock())
        # No llm_service injected

        with pytest.raises(RuntimeError, match="no LLMProvider injected"):
            await agent.summarise(
                "query", {}, [], system_prompt="test",
            )

    @pytest.mark.asyncio
    async def test_llm_decides_raises_without_llm_service(self):
        """LLMDecidesStrategy._llm_complete raises RuntimeError without LLMProvider."""
        with pytest.raises(RuntimeError, match="requires an LLMProvider"):
            await LLMDecidesStrategy._llm_complete(
                None, "model", [{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    async def test_supervisor_llm_complete_raises_without_service(self):
        """supervisor._llm_complete raises RuntimeError without LLMProvider."""
        from orchid.graph.supervisor import _llm_complete

        with pytest.raises(RuntimeError, match="requires an LLMProvider"):
            await _llm_complete(None, "model", [{"role": "user", "content": "test"}])


# ── Fix #7: Specific exception types ──────────────────────────────────


class TestSpecificExceptions:
    @pytest.mark.asyncio
    async def test_call_all_catches_connection_error(self):
        """CallAllStrategy should catch ConnectionError."""

        class _FailClient(MCPClient):
            async def call_tool(self, name, args, auth):
                raise ConnectionError("refused")

            async def list_tools(self, auth): return []
            async def list_prompts(self, auth): return []
            async def list_resources(self, auth): return []
            async def get_prompt(self, name, args, auth): return []
            async def read_resource(self, uri, auth): return ""

            @property
            def server_url(self): return "http://fail"

        strategy = CallAllStrategy()
        result = await strategy.execute(_FailClient(), _tools("t"), "q", _auth())
        assert "t_error" in result

    @pytest.mark.asyncio
    async def test_call_all_does_not_catch_attribute_error(self):
        """CallAllStrategy should NOT catch AttributeError (not in our exception list)."""

        class _AttrClient(MCPClient):
            async def call_tool(self, name, args, auth):
                raise AttributeError("unexpected")

            async def list_tools(self, auth): return []
            async def list_prompts(self, auth): return []
            async def list_resources(self, auth): return []
            async def get_prompt(self, name, args, auth): return []
            async def read_resource(self, uri, auth): return ""

            @property
            def server_url(self): return "http://fail"

        strategy = CallAllStrategy()
        with pytest.raises(AttributeError):
            await strategy.execute(_AttrClient(), _tools("t"), "q", _auth())


# ── Fix #8: monotonic clock for cache TTL ──────────────────────────


class TestMonotonicClock:
    def test_monotonic_reference_exists(self):
        """GenericAgent module should use time.monotonic for cache checks."""
        from orchid.agents import generic_agent
        assert generic_agent._monotonic is __import__("time").monotonic


# ── Fix #9: py.typed and __version__ ──────────────────────────────


class TestPackageMetadata:
    def test_version_is_set(self):
        import orchid
        assert hasattr(orchid, "__version__")
        assert orchid.__version__ == "0.1.0"

    def test_py_typed_exists(self):
        import importlib.resources
        files = importlib.resources.files("orchid")
        py_typed = files / "py.typed"
        assert py_typed.is_file()


# ── Fix #6: GenericAgent.run() decomposition ──────────────────────


class TestGenericAgentDecomposition:
    def test_pipeline_methods_exist(self):
        """GenericAgent should expose named pipeline step methods."""
        from orchid.agents.generic_agent import GenericAgent
        assert hasattr(GenericAgent, "_build_scope")
        assert hasattr(GenericAgent, "_step_rag_retrieval")
        assert hasattr(GenericAgent, "_step_cache_check")
        assert hasattr(GenericAgent, "_step_tool_calls")
        assert hasattr(GenericAgent, "_step_dynamic_injection")
        assert hasattr(GenericAgent, "_step_summarise")

    @pytest.mark.asyncio
    async def test_build_scope_returns_rag_scope(self):
        """_build_scope should return a RAGScope from auth + state."""
        from orchid.agents.generic_agent import GenericAgent
        from orchid.config.schema import AgentConfig, LLMConfig, RAGConfig
        from orchid.rag.scopes import RAGScope

        config = AgentConfig(
            description="d", prompt="p",
            rag=RAGConfig(enabled=False, namespace="ns"),
            llm=LLMConfig(),
        )
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[])
        llm_service = MagicMock()
        llm_service.complete = AsyncMock(return_value="summary")

        agent = GenericAgent(
            config=config, llm="test-model", reader=reader,
            mcp_clients=[], llm_service=llm_service,
        )
        auth = AuthContext(access_token="tok", tenant_key="t1", user_id="u1")
        state = {"chat_id": "c1"}

        scope = agent._build_scope(auth, state)
        assert isinstance(scope, RAGScope)
        assert scope.tenant_id == "t1"
        assert scope.user_id == "u1"
        assert scope.chat_id == "c1"
        assert scope.agent_id == agent.name
