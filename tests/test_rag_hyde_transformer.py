"""Tests for ``HyDETransformer`` (ADR-023)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.rag.transformers.hyde import HyDETransformer


def _resp(content: str) -> MagicMock:
    return MagicMock(content=content)


class TestPreStrategyFlag:
    def test_is_post_strategy(self):
        """HyDE fans out — must NOT be marked pre_strategy."""
        assert HyDETransformer.pre_strategy is False


class TestSingleHypothetical:
    @pytest.mark.asyncio
    async def test_returns_one_paragraph(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("Hypothetical answer paragraph about the topic."))
        result = await HyDETransformer().transform("What is X?", chat_model=chat_model)
        assert result == ["Hypothetical answer paragraph about the topic."]
        chat_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp(""))
        assert await HyDETransformer().transform("X?", chat_model=chat_model) == []


class TestMultipleHypothetical:
    @pytest.mark.asyncio
    async def test_returns_n_paragraphs(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("para one\npara two\npara three"))
        result = await HyDETransformer(n_hypothetical=3).transform("X?", chat_model=chat_model)
        assert result == ["para one", "para two", "para three"]

    @pytest.mark.asyncio
    async def test_caps_at_n(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("one\ntwo\nthree\nfour\nfive"))
        result = await HyDETransformer(n_hypothetical=2).transform("X?", chat_model=chat_model)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("alpha\n\n\nbeta\n   \n"))
        result = await HyDETransformer(n_hypothetical=3).transform("X?", chat_model=chat_model)
        assert result == ["alpha", "beta"]


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_no_chat_model_returns_empty_list(self):
        assert await HyDETransformer().transform("X?", chat_model=None) == []

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty_list(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        assert await HyDETransformer().transform("X?", chat_model=chat_model) == []


class TestValidation:
    def test_zero_n_hypothetical_raises(self):
        with pytest.raises(ValueError, match="n_hypothetical"):
            HyDETransformer(n_hypothetical=0)

    def test_negative_n_hypothetical_raises(self):
        with pytest.raises(ValueError, match="n_hypothetical"):
            HyDETransformer(n_hypothetical=-1)
