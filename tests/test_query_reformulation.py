"""Tests for query reformulation (OrchidAgent.reformulate_query)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.core.agent import OrchidAgent


class _TestAgent(OrchidAgent):
    @property
    def name(self):
        return "test"

    @property
    def description(self):
        return "test agent"

    async def run(self, state):
        return state


def _make_agent(*, chat_model=None):
    mock_reader = MagicMock()
    mock_reader.retrieve = AsyncMock(return_value=[])
    return _TestAgent(llm="test-model", reader=mock_reader, chat_model=chat_model)


def _make_state(query: str, history: list | None = None):
    messages = list(history or [])
    messages.append(HumanMessage(content=query))
    return {"messages": messages}


class TestReformulateQuery:
    @pytest.mark.asyncio
    async def test_reformulates_with_history(self):
        """Rewrites ambiguous query using conversation context."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="list vegan dishes"))

        agent = _make_agent(chat_model=chat_model)
        state = _make_state(
            "yes, show them",
            history=[
                HumanMessage(content="Do you have vegan options?"),
                AIMessage(content="Yes, we have several vegan dishes."),
            ],
        )

        result = await agent.reformulate_query("yes, show them", state)
        assert result == "list vegan dishes"
        chat_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_original_without_history(self):
        """No history → returns query unchanged."""
        chat_model = MagicMock()
        agent = _make_agent(chat_model=chat_model)
        state = _make_state("list vegan dishes")

        result = await agent.reformulate_query("list vegan dishes", state)
        assert result == "list vegan dishes"
        chat_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_without_chat_model(self):
        """No chat model → returns query unchanged."""
        agent = _make_agent(chat_model=None)
        state = _make_state(
            "yes",
            history=[
                HumanMessage(content="Any specials?"),
                AIMessage(content="Pan-Seared Duck."),
            ],
        )

        result = await agent.reformulate_query("yes", state)
        assert result == "yes"

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """LLM error → returns original query."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        agent = _make_agent(chat_model=chat_model)
        state = _make_state(
            "tell me more",
            history=[
                HumanMessage(content="What's the soup of the day?"),
                AIMessage(content="Tomato basil."),
            ],
        )

        result = await agent.reformulate_query("tell me more", state)
        assert result == "tell me more"

    @pytest.mark.asyncio
    async def test_fallback_on_empty_result(self):
        """Empty LLM response → returns original query."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content=""))

        agent = _make_agent(chat_model=chat_model)
        state = _make_state(
            "ok",
            history=[
                HumanMessage(content="Suggest something"),
                AIMessage(content="Try the pasta."),
            ],
        )

        result = await agent.reformulate_query("ok", state)
        assert result == "ok"
