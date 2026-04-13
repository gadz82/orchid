"""Tests for BaseAgent.summarise() — conversation history + prior tool context."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.agent import BaseAgent


# ── Concrete stub (BaseAgent is abstract) ──────────────────


class _StubAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "test stub"

    async def run(self, state: Any) -> Any:
        return {}


def _make_agent() -> _StubAgent:
    llm_service = MagicMock()
    llm_service.complete = AsyncMock(return_value="summary text")
    reader = MagicMock()
    return _StubAgent(llm="test-model", reader=reader, llm_service=llm_service)


# ── Tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarise_no_history_no_prior_context() -> None:
    """Baseline: summarise without history or prior context."""
    agent = _make_agent()
    result = await agent.summarise(
        "What courses?",
        {"tool1": "data"},
        [],
        system_prompt="You are an agent.",
    )
    assert result == "summary text"

    call_args = agent._llm_service.complete.call_args
    messages = call_args.args[1]

    # system + user only (no history injected)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # System prompt should NOT have the focus instruction
    assert "IMPORTANT:" not in messages[0]["content"]
    # System prompt should NOT have prior tool results
    assert "Previous Tool Results" not in messages[0]["content"]


@pytest.mark.asyncio
async def test_summarise_with_conversation_history() -> None:
    """History is injected between system and user messages."""
    agent = _make_agent()
    history = [
        {"role": "user", "content": "List notifications"},
        {"role": "assistant", "content": "Here are 9 notifications..."},
    ]

    await agent.summarise(
        "Show details",
        {},
        [],
        system_prompt="You are an agent.",
        conversation_history=history,
    )

    messages = agent._llm_service.complete.call_args.args[1]

    # system + 2 history + user = 4
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "List notifications"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"  # current query

    # Focus instruction should be appended to system prompt
    assert "IMPORTANT:" in messages[0]["content"]
    assert "LATEST message" in messages[0]["content"]


@pytest.mark.asyncio
async def test_summarise_with_prior_tool_context() -> None:
    """Prior tool results are appended to the system prompt."""
    agent = _make_agent()
    prior = {"create_notification": '{"id": "abc-123", "name": "Admin ILT"}'}

    await agent.summarise(
        "You forgot the template",
        {},
        [],
        system_prompt="You are an agent.",
        prior_tool_context=prior,
    )

    messages = agent._llm_service.complete.call_args.args[1]
    system_content = messages[0]["content"]

    assert "Previous Tool Results" in system_content
    assert "abc-123" in system_content
    assert "Admin ILT" in system_content


@pytest.mark.asyncio
async def test_summarise_with_both_history_and_prior_context() -> None:
    """Both features work together without interference."""
    agent = _make_agent()
    history = [
        {"role": "user", "content": "Create a notification"},
        {"role": "assistant", "content": "Created Admin ILT notification"},
    ]
    prior = {"create_notification": '{"id": "abc-123"}'}

    await agent.summarise(
        "Add a template",
        {},
        [],
        system_prompt="You are an agent.",
        conversation_history=history,
        prior_tool_context=prior,
    )

    messages = agent._llm_service.complete.call_args.args[1]

    # system + 2 history + user = 4
    assert len(messages) == 4
    system_content = messages[0]["content"]
    # Both features present in system prompt
    assert "IMPORTANT:" in system_content
    assert "Previous Tool Results" in system_content
    assert "abc-123" in system_content


@pytest.mark.asyncio
async def test_summarise_prior_context_truncated() -> None:
    """Large prior tool context is truncated to 4000 chars."""
    agent = _make_agent()
    # Create a large prior context
    prior = {"big_tool": "x" * 10000}

    await agent.summarise(
        "Query",
        {},
        [],
        system_prompt="Prompt.",
        prior_tool_context=prior,
    )

    system_content = agent._llm_service.complete.call_args.args[1][0]["content"]
    # The prior context section should exist but be capped
    assert "Previous Tool Results" in system_content
    # Total system prompt should not contain the full 10000 chars
    assert len(system_content) < 6000
