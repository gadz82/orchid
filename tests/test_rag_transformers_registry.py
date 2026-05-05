"""Tests for the query transformer registry (ADR-023)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from orchid_ai.core.retrieval import OrchidQueryTransformer, apply_pre_strategy
from orchid_ai.rag.transformers import (
    TRANSFORMER_REGISTRY,
    DecomposeTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    ReformulateTransformer,
    clear_query_transformers,
    get_query_transformer,
    register_query_transformer,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_query_transformers()
    yield
    clear_query_transformers()


class _CustomTransformer(OrchidQueryTransformer):
    pre_strategy: ClassVar[bool] = False

    async def transform(self, query, *, chat_model, history=None):
        return [f"custom:{query}"]


class _BadPreTransformer(OrchidQueryTransformer):
    """Marked pre_strategy=True but returns 0 results — must trigger the runtime check."""

    pre_strategy: ClassVar[bool] = True

    async def transform(self, query, *, chat_model, history=None):
        return []


class _MultiPreTransformer(OrchidQueryTransformer):
    """Marked pre_strategy=True but returns >1 — must also trigger the check."""

    pre_strategy: ClassVar[bool] = True

    async def transform(self, query, *, chat_model, history=None):
        return ["a", "b"]


class TestBuiltins:
    def test_reformulate_registered(self):
        assert "reformulate" in TRANSFORMER_REGISTRY
        assert isinstance(get_query_transformer("reformulate"), ReformulateTransformer)

    def test_multi_query_registered(self):
        assert "multi_query" in TRANSFORMER_REGISTRY
        assert isinstance(get_query_transformer("multi_query"), MultiQueryTransformer)

    def test_hyde_registered(self):
        assert "hyde" in TRANSFORMER_REGISTRY
        assert isinstance(get_query_transformer("hyde"), HyDETransformer)

    def test_decompose_registered(self):
        assert "decompose" in TRANSFORMER_REGISTRY
        assert isinstance(get_query_transformer("decompose"), DecomposeTransformer)


class TestPreStrategyFlags:
    def test_reformulate_is_pre_strategy(self):
        assert ReformulateTransformer.pre_strategy is True

    def test_multi_query_is_not_pre_strategy(self):
        assert MultiQueryTransformer.pre_strategy is False

    def test_hyde_is_not_pre_strategy(self):
        assert HyDETransformer.pre_strategy is False

    def test_decompose_is_not_pre_strategy(self):
        assert DecomposeTransformer.pre_strategy is False


class TestRegistration:
    def test_register_and_get(self):
        register_query_transformer("custom", _CustomTransformer)
        assert isinstance(get_query_transformer("custom"), _CustomTransformer)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown query transformer"):
            get_query_transformer("nope")


class TestApplyPreStrategy:
    @pytest.mark.asyncio
    async def test_runs_pre_strategy_in_order(self):
        class _Append(OrchidQueryTransformer):
            pre_strategy: ClassVar[bool] = True

            def __init__(self, suffix):
                self._suffix = suffix

            async def transform(self, query, *, chat_model, history=None):
                return [f"{query}{self._suffix}"]

        result = await apply_pre_strategy([_Append("-A"), _Append("-B")], "raw", chat_model=None)
        assert result == "raw-A-B"

    @pytest.mark.asyncio
    async def test_skips_non_pre_strategy(self):
        result = await apply_pre_strategy([_CustomTransformer()], "raw", chat_model=None)
        assert result == "raw"

    @pytest.mark.asyncio
    async def test_zero_result_raises(self):
        with pytest.raises(RuntimeError, match="exactly 1"):
            await apply_pre_strategy([_BadPreTransformer()], "raw", chat_model=None)

    @pytest.mark.asyncio
    async def test_multi_result_raises(self):
        with pytest.raises(RuntimeError, match="exactly 1"):
            await apply_pre_strategy([_MultiPreTransformer()], "raw", chat_model=None)
