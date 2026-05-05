"""Tests for ``ReformulateTransformer`` (replaces the legacy
``OrchidAgent.reformulate_query`` method)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.rag.transformers.reformulate import ReformulateTransformer


def _history(*pairs):
    """Build a [{role, content}, ...] list from alternating user/assistant pairs."""
    out = []
    for i, content in enumerate(pairs):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    return out


class TestReformulateTransformer:
    @pytest.mark.asyncio
    async def test_pre_strategy_flag_is_true(self):
        """Reformulate is the canonical pre_strategy transformer."""
        assert ReformulateTransformer.pre_strategy is True

    @pytest.mark.asyncio
    async def test_reformulates_with_history(self):
        """Rewrites ambiguous query using conversation context."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="list vegan dishes"))

        result = await ReformulateTransformer().transform(
            "yes, show them",
            chat_model=chat_model,
            history=_history("Do you have vegan options?", "Yes, we have several vegan dishes."),
        )
        assert result == ["list vegan dishes"]
        chat_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_original_without_history(self):
        """No history → returns query unchanged in a 1-element list."""
        chat_model = MagicMock()
        result = await ReformulateTransformer().transform("list vegan dishes", chat_model=chat_model, history=None)
        assert result == ["list vegan dishes"]
        chat_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_without_chat_model(self):
        """No chat model → returns query unchanged."""
        result = await ReformulateTransformer().transform(
            "yes",
            chat_model=None,
            history=_history("Any specials?", "Pan-Seared Duck."),
        )
        assert result == ["yes"]

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """LLM error → returns original query."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        result = await ReformulateTransformer().transform(
            "tell me more",
            chat_model=chat_model,
            history=_history("What's the soup of the day?", "Tomato basil."),
        )
        assert result == ["tell me more"]

    @pytest.mark.asyncio
    async def test_fallback_on_empty_result(self):
        """Empty LLM response → returns original query."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content=""))

        result = await ReformulateTransformer().transform(
            "ok",
            chat_model=chat_model,
            history=_history("Suggest something", "Try the pasta."),
        )
        assert result == ["ok"]

    @pytest.mark.asyncio
    async def test_always_returns_single_element_list(self):
        """The pre_strategy contract — ReformulateTransformer never returns ≠ 1."""
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="one short rewrite"))

        result = await ReformulateTransformer().transform("raw", chat_model=chat_model, history=_history("Q", "A"))
        assert isinstance(result, list)
        assert len(result) == 1
