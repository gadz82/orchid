"""Tests for ``DecomposeTransformer`` (ADR-023)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.rag.transformers.decompose import DecomposeTransformer


def _resp(content: str) -> MagicMock:
    return MagicMock(content=content)


class TestPreStrategyFlag:
    def test_is_post_strategy(self):
        """Decompose fans out — must NOT be marked pre_strategy."""
        assert DecomposeTransformer.pre_strategy is False


class TestDecompose:
    @pytest.mark.asyncio
    async def test_returns_sub_queries(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("Sub Q one\nSub Q two\nSub Q three"))
        result = await DecomposeTransformer().transform("Compare A, B, and C", chat_model=chat_model)
        assert result == ["Sub Q one", "Sub Q two", "Sub Q three"]

    @pytest.mark.asyncio
    async def test_atomic_query_returns_single_line(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("What is the capital of France?"))
        result = await DecomposeTransformer().transform("What is the capital of France?", chat_model=chat_model)
        assert result == ["What is the capital of France?"]

    @pytest.mark.asyncio
    async def test_caps_at_max_sub_queries(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("one\ntwo\nthree\nfour\nfive\nsix"))
        result = await DecomposeTransformer(max_sub_queries=3).transform("X", chat_model=chat_model)
        assert len(result) == 3
        assert result == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_resp("alpha\n\nbeta\n  \n\ngamma"))
        result = await DecomposeTransformer().transform("X", chat_model=chat_model)
        assert result == ["alpha", "beta", "gamma"]


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_no_chat_model_returns_empty_list(self):
        assert await DecomposeTransformer().transform("X", chat_model=None) == []

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty_list(self):
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        assert await DecomposeTransformer().transform("X", chat_model=chat_model) == []


class TestValidation:
    def test_max_sub_queries_must_be_at_least_2(self):
        with pytest.raises(ValueError, match="max_sub_queries"):
            DecomposeTransformer(max_sub_queries=1)
