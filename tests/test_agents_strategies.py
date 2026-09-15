"""Tests for src.agents.strategies — OrchidToolCallStrategy implementations."""

from __future__ import annotations

import pytest

from orchid_ai.agents.strategies import (
    STRATEGY_REGISTRY,
    CallAllStrategy,
    LLMDecidesStrategy,
    SequentialStrategy,
    get_strategy,
)
from orchid_ai.config.schema import OrchidToolConfig
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext

# ── Inline mock MCP client ──────────────────────────────────────


class _MockMCPClient(OrchidMCPClient):
    """Lightweight mock for strategy tests."""

    def __init__(self, results=None, raise_for=None):
        self._results = results or {}
        self._raise_for = raise_for or set()
        self.calls: list[dict] = []

    async def call_tool(self, tool_name, arguments, auth):
        self.calls.append({"tool": tool_name, "args": arguments})
        if tool_name in self._raise_for:
            raise RuntimeError(f"Tool {tool_name} failed")
        if tool_name in self._results:
            return self._results[tool_name]
        return OrchidMCPToolResult(content=[{"type": "text", "text": f"result_{tool_name}"}])

    async def list_tools(self, auth):
        return [{"name": k, "description": f"Tool {k}"} for k in self._results]

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
        return "http://mock"


# ── Helpers ─────────────────────────────────────────────────────


def _auth():
    return OrchidAuthContext(access_token="test-token")


def _tools(*names):
    return [OrchidToolConfig(name=n) for n in names]


# ── CallAllStrategy ─────────────────────────────────────────────


class TestCallAllStrategy:
    @pytest.mark.asyncio
    async def test_calls_every_tool(self):
        client = _MockMCPClient()
        strategy = CallAllStrategy()
        tools = _tools("tool_a", "tool_b")
        results = await strategy.execute(client, tools, "query", _auth())
        assert len(client.calls) == 2
        assert "tool_a" in results
        assert "tool_b" in results

    @pytest.mark.asyncio
    async def test_returns_results_keyed_by_name(self):
        client = _MockMCPClient(
            results={
                "tool_a": OrchidMCPToolResult(content=[{"type": "text", "text": "hello"}]),
            }
        )
        strategy = CallAllStrategy()
        results = await strategy.execute(client, _tools("tool_a"), "q", _auth())
        assert results["tool_a"] == "hello"

    @pytest.mark.asyncio
    async def test_handles_tool_errors(self):
        client = _MockMCPClient(raise_for={"tool_a"})
        strategy = CallAllStrategy()
        results = await strategy.execute(client, _tools("tool_a", "tool_b"), "q", _auth())
        assert "tool_a_error" in results
        assert "tool_b" in results


# ── SequentialStrategy ──────────────────────────────────────────


class TestSequentialStrategy:
    @pytest.mark.asyncio
    async def test_calls_tools_in_order(self):
        client = _MockMCPClient()
        strategy = SequentialStrategy()
        tools = _tools("first", "second", "third")
        await strategy.execute(client, tools, "q", _auth())
        call_order = [c["tool"] for c in client.calls]
        assert call_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_passes_previous_results(self):
        client = _MockMCPClient()
        strategy = SequentialStrategy()
        tools = _tools("tool_a", "tool_b")
        await strategy.execute(client, tools, "q", _auth())
        # Second call should have previous_results in args
        second_call = client.calls[1]
        assert "previous_results" in second_call["args"]

    @pytest.mark.asyncio
    async def test_first_call_has_no_previous_results(self):
        client = _MockMCPClient()
        strategy = SequentialStrategy()
        tools = _tools("tool_a", "tool_b")
        await strategy.execute(client, tools, "q", _auth())
        first_call = client.calls[0]
        assert "previous_results" not in first_call["args"]

    @pytest.mark.asyncio
    async def test_handles_errors_gracefully(self):
        client = _MockMCPClient(raise_for={"tool_a"})
        strategy = SequentialStrategy()
        tools = _tools("tool_a", "tool_b")
        results = await strategy.execute(client, tools, "q", _auth())
        assert "tool_a_error" in results
        # tool_b should still be called
        assert "tool_b" in results


# ── get_strategy ────────────────────────────────────────────────


class TestGetStrategy:
    def test_all_returns_call_all(self):
        assert isinstance(get_strategy("all"), CallAllStrategy)

    def test_sequential_returns_sequential(self):
        assert isinstance(get_strategy("sequential"), SequentialStrategy)

    def test_llm_decides_returns_llm_decides(self):
        assert isinstance(get_strategy("llm_decides"), LLMDecidesStrategy)

    def test_unknown_falls_back_to_call_all(self):
        assert isinstance(get_strategy("nonexistent"), CallAllStrategy)


# ── STRATEGY_REGISTRY ───────────────────────────────────────────


class TestStrategyRegistry:
    def test_has_all_key(self):
        assert "all" in STRATEGY_REGISTRY

    def test_has_sequential_key(self):
        assert "sequential" in STRATEGY_REGISTRY

    def test_has_llm_decides_key(self):
        assert "llm_decides" in STRATEGY_REGISTRY

    def test_values_are_strategy_subclasses(self):
        from orchid_ai.agents.strategies import OrchidToolCallStrategy

        for cls in STRATEGY_REGISTRY.values():
            assert issubclass(cls, OrchidToolCallStrategy)
