"""Tests for GenericAgent multi-turn context (history + prior tool results)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import AgentConfig, LLMConfig, RAGConfig
from orchid_ai.core.state import AuthContext


def _make_agent() -> GenericAgent:
    config = AgentConfig(
        description="Test agent",
        prompt="You are a test agent.",
        rag=RAGConfig(enabled=False, namespace="test"),
        llm=LLMConfig(model="test-model"),
    )
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="summary response"))
    return GenericAgent(
        config=config,
        llm="test-model",
        reader=reader,
        mcp_clients=[],
        chat_model=chat_model,
    )


def _make_state(
    query: str = "follow-up question",
    *,
    history: list[Any] | None = None,
    mcp_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = history or []
    messages.append(HumanMessage(content=query))
    state: dict[str, Any] = {
        "messages": messages,
        "auth_context": AuthContext(
            access_token="tok",
            tenant_key="t1",
            user_id="u1",
        ),
        "chat_id": "chat-1",
    }
    if mcp_context is not None:
        state["mcp_context"] = mcp_context
    return state


@pytest.mark.asyncio
async def test_summarise_receives_conversation_history() -> None:
    """GenericAgent passes conversation history to summarise()."""
    agent = _make_agent()

    state = _make_state(
        "tell me more",
        history=[
            HumanMessage(content="What courses are available?"),
            AIMessage(content="Here are 5 courses..."),
        ],
    )

    with patch.object(agent._mcp_dispatcher, "fetch", new_callable=AsyncMock, return_value={}):
        await agent.run(state)

    # Check that ainvoke was called with conversation_history
    call_args = agent._chat_model.ainvoke.call_args
    messages = call_args.args[0]

    # Should have: system + 2 history + user = 4 messages
    assert len(messages) == 4
    assert messages[1]["role"] == "user"
    assert "What courses" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert "5 courses" in messages[2]["content"]


@pytest.mark.asyncio
async def test_summarise_receives_prior_tool_context() -> None:
    """GenericAgent passes prior mcp_context to summarise()."""
    agent = _make_agent()

    # Agent name from config is "" (default before _apply_defaults sets it)
    # Override for test clarity
    agent._config.name = "learning"

    state = _make_state(
        "tell me more about the first one",
        mcp_context={"learning": {"search_courses": '{"results": [{"id": 1, "name": "Python 101"}]}'}},
    )

    with patch.object(agent._mcp_dispatcher, "fetch", new_callable=AsyncMock, return_value={}):
        await agent.run(state)

    system_content = agent._chat_model.ainvoke.call_args.args[0][0]["content"]
    assert "Previous Tool Results" in system_content
    assert "Python 101" in system_content


@pytest.mark.asyncio
async def test_no_prior_context_when_mcp_context_empty() -> None:
    """When mcp_context has no data for this agent, no prior context is injected."""
    agent = _make_agent()

    state = _make_state("hello")

    with patch.object(agent._mcp_dispatcher, "fetch", new_callable=AsyncMock, return_value={}):
        await agent.run(state)

    system_content = agent._chat_model.ainvoke.call_args.args[0][0]["content"]
    assert "Previous Tool Results" not in system_content


@pytest.mark.asyncio
async def test_no_prior_context_for_different_agent() -> None:
    """Prior tool context from a DIFFERENT agent is not injected."""
    agent = _make_agent()
    agent._config.name = "learning"

    state = _make_state(
        "query",
        mcp_context={"notifications": {"list_notifications": "some data"}},
    )

    with patch.object(agent._mcp_dispatcher, "fetch", new_callable=AsyncMock, return_value={}):
        await agent.run(state)

    system_content = agent._chat_model.ainvoke.call_args.args[0][0]["content"]
    # Should NOT contain notifications data — it belongs to a different agent
    assert "Previous Tool Results" not in system_content
