"""Tests for ``MultiQueryRetrieval`` (ADR-023)."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.retrieval import OrchidQueryTransformer
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.strategies.multi_query import MultiQueryRetrieval


class _ExternalTransformer(OrchidQueryTransformer):
    """Test double — adds a fixed extra query to the fan-out set."""

    pre_strategy: ClassVar[bool] = False

    async def transform(self, query, *, chat_model, history=None):
        return [f"{query}-external"]


class _PreStrategyDouble(OrchidQueryTransformer):
    """A pre_strategy=True transformer that the strategy must skip."""

    pre_strategy: ClassVar[bool] = True

    async def transform(self, query, *, chat_model, history=None):  # pragma: no cover
        raise AssertionError("strategy must not call pre_strategy transformers")


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _result(doc_id: str, score: float) -> OrchidSearchResult:
    return OrchidSearchResult(
        document=Document(id=doc_id, page_content=doc_id, metadata={}),
        score=score,
    )


class TestMultiQueryRetrieval:
    @pytest.mark.asyncio
    async def test_merges_results_from_variations(self):
        """Both queries (original + 1 variation) feed retrieval; output is deduped."""
        reader = MagicMock()
        call_count = 0

        async def _retrieve(query, namespace, k, scope):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_result(f"doc-{i}", 1.0 - i * 0.1) for i in range(3)]
            return [_result(f"doc-{i}", 0.8 - (i - 2) * 0.1) for i in range(2, 5)]

        reader.retrieve = AsyncMock(side_effect=_retrieve)

        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="variation 1"))

        results = await MultiQueryRetrieval(num_queries=1).retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )

        ids = {r.document.id for r in results}
        assert len(results) <= 5
        assert len(ids) == len(results)  # deduplicated by id

    @pytest.mark.asyncio
    async def test_dedupe_keeps_highest_score(self):
        reader = MagicMock()
        reader.retrieve = AsyncMock(
            side_effect=[
                [_result("d1", 0.5)],
                [_result("d1", 0.9)],
            ]
        )
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="v"))

        results = await MultiQueryRetrieval(num_queries=1).retrieve(
            query="q", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        assert len(results) == 1
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_falls_back_to_single_query_without_chat_model(self):
        """No chat_model → only the original query runs (no fan-out)."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("d", 0.5)])

        results = await MultiQueryRetrieval().retrieve(
            query="q", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=None
        )
        assert len(results) == 1
        assert reader.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_variation_failure_continues_with_original(self):
        """LLM error during variation generation → fall back to single-query retrieval."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("d", 0.5)])

        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        results = await MultiQueryRetrieval().retrieve(
            query="q", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_external_transformers_compose_with_internal(self):
        """External transformers fan out alongside the internal MultiQueryTransformer."""
        reader = MagicMock()
        # 1 original + 1 internal-multi-query variation + 1 external = 3 retrieves.
        reader.retrieve = AsyncMock(
            side_effect=[
                [_result("a", 0.4)],
                [_result("b", 0.6)],
                [_result("c", 0.8)],
            ]
        )
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="variation_one"))

        results = await MultiQueryRetrieval(num_queries=1).retrieve(
            query="orig",
            namespace="kb",
            scope=_scope(),
            k=5,
            reader=reader,
            chat_model=chat_model,
            transformers=[_ExternalTransformer()],
        )
        assert reader.retrieve.await_count == 3
        ids = {r.document.id for r in results}
        assert ids == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_pre_strategy_transformers_skipped(self):
        """Pre_strategy transformers in the list never run inside the strategy."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(side_effect=[[_result("a", 0.5)], [_result("b", 0.7)]])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="v"))

        results = await MultiQueryRetrieval(num_queries=1).retrieve(
            query="orig",
            namespace="kb",
            scope=_scope(),
            k=5,
            reader=reader,
            chat_model=chat_model,
            transformers=[_PreStrategyDouble()],
        )
        # Original + 1 internal variation = 2; pre_strategy double skipped.
        assert reader.retrieve.await_count == 2
        assert len(results) == 2
