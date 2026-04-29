"""Tests for GenericAgent dynamic RAG injection filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMCPServerConfig,
    OrchidRAGConfig,
    OrchidToolConfig,
)
from orchid_ai.core.state import OrchidAuthContext


def _make_state(query: str = "test query") -> dict[str, Any]:
    return {
        "messages": [MagicMock(content=query)],
        "auth_context": OrchidAuthContext(
            access_token="tok",
            tenant_key="t1",
            user_id="u1",
        ),
        "chat_id": "chat-1",
    }


def _make_agent(config: OrchidAgentConfig) -> GenericAgent:
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="summary"))
    return GenericAgent(
        config=config,
        model_id="test-model",
        reader=reader,
        mcp_clients=[],
        chat_model=chat_model,
    )


@pytest.mark.asyncio
async def test_no_injection_when_no_injectable_tools():
    """When no tools have inject_to_rag=True, inject_to_rag() is never called."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns"),
        llm=OrchidLLMConfig(),
    )
    assert config.injectable_tools == set()

    agent = _make_agent(config)

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {}),
        ),
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock) as mock_inject,
    ):
        await agent.run(_make_state())
        mock_inject.assert_not_called()


@pytest.mark.asyncio
async def test_injection_only_for_opted_in_mcp_tools():
    """Only MCP tool results with inject_to_rag=True are passed to inject_to_rag()."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns"),
        llm=OrchidLLMConfig(),
        mcp_servers=[
            OrchidMCPServerConfig(
                name="srv",
                url="http://x",
                tools=[
                    OrchidToolConfig(name="tool_keep", inject_to_rag=True),
                    OrchidToolConfig(name="tool_skip"),
                ],
            ),
        ],
    )
    config.injectable_tools = {"tool_keep"}

    agent = _make_agent(config)

    mcp_results = {"tool_keep": "data1", "tool_skip": "data2"}

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, mcp_results),
        ),
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock) as mock_inject,
    ):
        await agent.run(_make_state())
        mock_inject.assert_called_once()
        injected_data = mock_inject.call_args.kwargs["mcp_data"]
        assert "tool_keep" in injected_data
        assert "tool_skip" not in injected_data


@pytest.mark.asyncio
async def test_injection_only_for_opted_in_builtin_tools():
    """Only built-in tool results with inject_to_rag=True are passed to inject_to_rag()."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns"),
        llm=OrchidLLMConfig(),
        tools=["format_date", "calc_rate"],
    )
    config.injectable_tools = {"builtin_format_date"}

    agent = _make_agent(config)

    tool_results = {"builtin_format_date": "2024-01-01", "builtin_calc_rate": "85%"}

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, tool_results),
        ),
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock) as mock_inject,
    ):
        await agent.run(_make_state())
        mock_inject.assert_called_once()
        injected_data = mock_inject.call_args.kwargs["mcp_data"]
        assert "builtin_format_date" in injected_data
        assert "builtin_calc_rate" not in injected_data


@pytest.mark.asyncio
async def test_no_injection_when_rag_disabled():
    """Even with injectable_tools set, injection is skipped when rag.enabled=False."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=False, namespace="ns"),
        llm=OrchidLLMConfig(),
        mcp_servers=[
            OrchidMCPServerConfig(
                name="srv",
                url="http://x",
                tools=[OrchidToolConfig(name="tool_a", inject_to_rag=True)],
            ),
        ],
    )
    config.injectable_tools = {"tool_a"}

    agent = _make_agent(config)

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"tool_a": "data"}),
        ),
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock) as mock_inject,
    ):
        await agent.run(_make_state())
        mock_inject.assert_not_called()


# ── Cache hit / skip tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_skips_tool_in_agentic_loop():
    """When cache has valid data for a tool, that tool is skipped in the agentic loop."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns", rag_ttl=3600),
        llm=OrchidLLMConfig(),
        mcp_servers=[
            OrchidMCPServerConfig(
                name="srv",
                url="http://x",
                tools=[
                    OrchidToolConfig(name="cached_tool", inject_to_rag=True),
                    OrchidToolConfig(name="fresh_tool"),
                ],
            ),
        ],
    )
    config.injectable_tools = {"cached_tool"}
    config.injectable_tool_ttls = {"cached_tool": 3600}

    agent = _make_agent(config)
    agent.reader.lookup_cached_tool_results = AsyncMock(return_value="cached data")

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"fresh_tool": "fresh data"}),
        ) as mock_loop,
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock),
    ):
        await agent.run(_make_state())
        mock_loop.assert_called_once()
        # skip_tools should contain the cached tool
        assert "cached_tool" in mock_loop.call_args.kwargs["skip_tools"]


@pytest.mark.asyncio
async def test_cache_miss_calls_tool_normally():
    """When cache returns None, the tool is called normally."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns", rag_ttl=3600),
        llm=OrchidLLMConfig(),
        mcp_servers=[
            OrchidMCPServerConfig(
                name="srv",
                url="http://x",
                tools=[OrchidToolConfig(name="tool_a", inject_to_rag=True)],
            ),
        ],
    )
    config.injectable_tools = {"tool_a"}
    config.injectable_tool_ttls = {"tool_a": 3600}

    agent = _make_agent(config)
    agent.reader.lookup_cached_tool_results = AsyncMock(return_value=None)

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"tool_a": "fresh"}),
        ) as mock_loop,
        patch("orchid_ai.agents.generic_agent.inject_to_rag", new_callable=AsyncMock),
    ):
        await agent.run(_make_state())
        # skip_tools should be empty (no cache hits)
        assert mock_loop.call_args.kwargs["skip_tools"] == set()


@pytest.mark.asyncio
async def test_no_cache_check_when_no_ttls():
    """When injectable_tool_ttls is empty, no cache lookup happens."""
    config = OrchidAgentConfig(
        description="d",
        prompt="p",
        rag=OrchidRAGConfig(enabled=True, namespace="ns"),
        llm=OrchidLLMConfig(),
    )
    assert config.injectable_tool_ttls == {}

    agent = _make_agent(config)
    agent.reader.lookup_cached_tool_results = AsyncMock()

    with patch.object(
        agent,
        "_agentic_tool_loop",
        new_callable=AsyncMock,
        return_value=(None, {}),
    ):
        await agent.run(_make_state())
        agent.reader.lookup_cached_tool_results.assert_not_called()
