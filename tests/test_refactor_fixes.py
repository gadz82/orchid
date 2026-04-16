"""Tests for the 10-point refactor — validates all behavioral changes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.strategies import CallAllStrategy, LLMDecidesStrategy
from orchid_ai.config.schema import ToolConfig
from orchid_ai.core.mcp import MCPClient, MCPToolResult
from orchid_ai.core.state import AuthContext


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
        from orchid_ai.core.agent import BaseAgent
        from orchid_ai.rag.scopes import RAGScope

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


# ── Fix #5: litellm fallback removed — RuntimeError on missing BaseChatModel ──


class TestNoLitellmFallback:
    @pytest.mark.asyncio
    async def test_summarise_raises_without_chat_model(self):
        """BaseAgent.summarise() should raise RuntimeError when no BaseChatModel injected."""
        from orchid_ai.core.agent import BaseAgent

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
        # No chat_model injected

        with pytest.raises(RuntimeError, match="no chat model injected"):
            await agent.summarise(
                "query",
                {},
                [],
                system_prompt="test",
            )

    @pytest.mark.asyncio
    async def test_llm_decides_raises_without_chat_model(self):
        """LLMDecidesStrategy._llm_complete raises RuntimeError without BaseChatModel."""
        with pytest.raises(RuntimeError, match="requires a BaseChatModel"):
            await LLMDecidesStrategy._llm_complete(
                None,
                "model",
                [{"role": "user", "content": "test"}],
            )

    @pytest.mark.asyncio
    async def test_supervisor_llm_complete_raises_without_service(self):
        """supervisor._llm_complete raises RuntimeError without BaseChatModel."""
        from orchid_ai.graph.supervisor import _llm_complete

        with pytest.raises(RuntimeError, match="requires a BaseChatModel"):
            await _llm_complete(None, "model", [{"role": "user", "content": "test"}])


# ── Fix #7: Specific exception types ──────────────────────────────────


class TestSpecificExceptions:
    @pytest.mark.asyncio
    async def test_call_all_catches_connection_error(self):
        """CallAllStrategy should catch ConnectionError."""

        class _FailClient(MCPClient):
            async def call_tool(self, name, args, auth):
                raise ConnectionError("refused")

            async def list_tools(self, auth):
                return []

            async def list_prompts(self, auth):
                return []

            async def list_resources(self, auth):
                return []

            async def get_prompt(self, name, args, auth):
                return []

            async def read_resource(self, uri, auth):
                return ""

            @property
            def server_url(self):
                return "http://fail"

        strategy = CallAllStrategy()
        result = await strategy.execute(_FailClient(), _tools("t"), "q", _auth())
        assert "t_error" in result

    @pytest.mark.asyncio
    async def test_call_all_catches_attribute_error_gracefully(self):
        """CallAllStrategy catches any Exception (including AttributeError) at the MCP boundary.

        MCP servers can fail with arbitrary exceptions (HTTP errors, protocol
        errors, buggy implementations).  The strategy is a fault-isolation
        boundary — it must degrade gracefully, not crash the agent.
        """

        class _AttrClient(MCPClient):
            async def call_tool(self, name, args, auth):
                raise AttributeError("unexpected")

            async def list_tools(self, auth):
                return []

            async def list_prompts(self, auth):
                return []

            async def list_resources(self, auth):
                return []

            async def get_prompt(self, name, args, auth):
                return []

            async def read_resource(self, uri, auth):
                return ""

            @property
            def server_url(self):
                return "http://fail"

        strategy = CallAllStrategy()
        result = await strategy.execute(_AttrClient(), _tools("t"), "q", _auth())
        assert "t_error" in result
        assert "unexpected" in result["t_error"]

    @pytest.mark.asyncio
    async def test_call_all_catches_http_status_error(self):
        """CallAllStrategy catches httpx.HTTPStatusError (e.g. 401 Unauthorized).

        This was the root cause of the infinite-retry crash when MCP servers
        returned HTTP 401 — the old exception tuple did not include it.
        """
        import httpx

        class _HttpErrorClient(MCPClient):
            async def call_tool(self, name, args, auth):
                response = httpx.Response(401, request=httpx.Request("POST", "http://mcp/tool"))
                raise httpx.HTTPStatusError("401 Unauthorized", request=response.request, response=response)

            async def list_tools(self, auth):
                return []

            async def list_prompts(self, auth):
                return []

            async def list_resources(self, auth):
                return []

            async def get_prompt(self, name, args, auth):
                return []

            async def read_resource(self, uri, auth):
                return ""

            @property
            def server_url(self):
                return "http://mcp"

        strategy = CallAllStrategy()
        result = await strategy.execute(_HttpErrorClient(), _tools("t"), "q", _auth())
        assert "t_error" in result
        assert "401" in result["t_error"]


# ── Fix #8: wall clock for cache TTL (matches dynamic.py) ──────────


class TestWallClock:
    def test_wall_clock_reference_exists(self):
        """GenericAgent module should use time.time for cache checks (matches dynamic.py)."""
        from orchid_ai.agents import generic_agent

        assert generic_agent._wall_clock is __import__("time").time


# ── Fix #9: py.typed and __version__ ──────────────────────────────


class TestPackageMetadata:
    def test_version_is_set(self):
        import orchid_ai

        assert hasattr(orchid_ai, "__version__")
        assert orchid_ai.__version__  # non-empty version string

    def test_py_typed_exists(self):
        import importlib.resources

        files = importlib.resources.files("orchid_ai")
        py_typed = files / "py.typed"
        assert py_typed.is_file()


# ── Fix #6: GenericAgent.run() decomposition ──────────────────────


class TestGenericAgentDecomposition:
    def test_pipeline_methods_exist(self):
        """GenericAgent should expose named pipeline step methods."""
        from orchid_ai.agents.generic_agent import GenericAgent

        assert hasattr(GenericAgent, "_build_scope")
        assert hasattr(GenericAgent, "_step_rag_retrieval")
        assert hasattr(GenericAgent, "_step_cache_check")
        assert hasattr(GenericAgent, "_agentic_tool_loop")
        assert hasattr(GenericAgent, "_step_dynamic_injection")
        assert hasattr(GenericAgent, "_step_summarise")

    @pytest.mark.asyncio
    async def test_build_scope_returns_rag_scope(self):
        """_build_scope should return a RAGScope from auth + state."""
        from orchid_ai.agents.generic_agent import GenericAgent
        from orchid_ai.config.schema import AgentConfig, LLMConfig, RAGConfig
        from orchid_ai.rag.scopes import RAGScope

        config = AgentConfig(
            description="d",
            prompt="p",
            rag=RAGConfig(enabled=False, namespace="ns"),
            llm=LLMConfig(),
        )
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="summary"))

        agent = GenericAgent(
            config=config,
            llm="test-model",
            reader=reader,
            mcp_clients=[],
            chat_model=chat_model,
        )
        auth = AuthContext(access_token="tok", tenant_key="t1", user_id="u1")
        state = {"chat_id": "c1"}

        scope = agent._build_scope(auth, state)
        assert isinstance(scope, RAGScope)
        assert scope.tenant_id == "t1"
        assert scope.user_id == "u1"
        assert scope.chat_id == "c1"
        assert scope.agent_id == agent.name


# ── render_capabilities HTTP error resilience ─────────────────────


class TestRenderCapabilitiesResilience:
    """MCPDispatcher.render_capabilities must degrade gracefully on HTTP errors."""

    @pytest.mark.asyncio
    async def test_render_capabilities_catches_http_status_error(self):
        """401/403/500 from an MCP server should be logged, not crash the agent."""
        import httpx

        from orchid_ai.agents.mcp_dispatcher import MCPDispatcher
        from orchid_ai.config.schema import MCPServerConfig

        class _Http401Client(MCPClient):
            async def call_tool(self, name, args, auth):
                return MCPToolResult(text="")

            async def list_tools(self, auth):
                response = httpx.Response(401, request=httpx.Request("GET", "http://mcp/tools"))
                raise httpx.HTTPStatusError("401 Unauthorized", request=response.request, response=response)

            async def list_prompts(self, auth):
                return []

            async def list_resources(self, auth):
                return []

            async def get_prompt(self, name, args, auth):
                return []

            async def read_resource(self, uri, auth):
                return ""

            @property
            def server_url(self):
                return "http://mcp"

        server_cfg = MCPServerConfig(name="failing-server", url="http://mcp", discover_all_tools=True)
        dispatcher = MCPDispatcher(mcp_clients=[_Http401Client()], server_configs=[server_cfg])

        # Should NOT raise — the 401 is caught and the server is skipped
        caps = await dispatcher.render_capabilities(_auth(), agent_name="test")
        assert caps.raw_tools == []
        assert caps.tool_client_map == {}

    @pytest.mark.asyncio
    async def test_fetch_catches_http_status_error(self):
        """MCPDispatcher.fetch should catch HTTP errors from MCP servers."""
        import httpx

        from orchid_ai.agents.mcp_dispatcher import MCPDispatcher
        from orchid_ai.config.schema import MCPServerConfig, ToolConfig as TC

        class _Http500Client(MCPClient):
            async def call_tool(self, name, args, auth):
                response = httpx.Response(500, request=httpx.Request("POST", "http://mcp/tool"))
                raise httpx.HTTPStatusError("500 Server Error", request=response.request, response=response)

            async def list_tools(self, auth):
                return []

            async def list_prompts(self, auth):
                return []

            async def list_resources(self, auth):
                return []

            async def get_prompt(self, name, args, auth):
                return []

            async def read_resource(self, uri, auth):
                return ""

            @property
            def server_url(self):
                return "http://mcp"

        server_cfg = MCPServerConfig(
            name="failing-server",
            url="http://mcp",
            tools=[TC(name="my_tool")],
            tool_call_strategy="all",
        )
        dispatcher = MCPDispatcher(mcp_clients=[_Http500Client()], server_configs=[server_cfg])

        # Should NOT raise — the error is caught and reported
        result = await dispatcher.fetch("test query", _auth(), agent_name="test")
        assert "my_tool_error" in result
