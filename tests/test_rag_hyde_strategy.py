"""Tests for ``HyDERetrieval``."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.config.schema import OrchidRetrievalConfig
from orchid_ai.config.schema_rag import OrchidHydeConfig
from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.retrieval import OrchidQueryTransformer
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.strategies.hyde import HyDERetrieval


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _result(doc_id: str, score: float) -> OrchidSearchResult:
    return OrchidSearchResult(
        document=Document(id=doc_id, page_content=doc_id, metadata={}),
        score=score,
    )


def _hyde_response(text: str) -> MagicMock:
    return MagicMock(content=text)


class _ExternalTransformer(OrchidQueryTransformer):
    """Test double — appends a fixed suffix as the only fan-out query."""

    pre_strategy: ClassVar[bool] = False

    async def transform(self, query, *, chat_model, history=None):
        return [f"{query}-external"]


class _PreStrategyDouble(OrchidQueryTransformer):
    """A pre_strategy=True transformer must be ignored by the strategy."""

    pre_strategy: ClassVar[bool] = True

    async def transform(self, query, *, chat_model, history=None):  # pragma: no cover
        raise AssertionError("HyDERetrieval must not call pre_strategy transformers")


class TestHydeRetrieval:
    @pytest.mark.asyncio
    async def test_fans_out_original_plus_hypothetical(self):
        reader = MagicMock()
        # Three queries land at the reader: original + 1 hypothetical
        # (default n_hypothetical=1).
        reader.retrieve = AsyncMock(side_effect=[[_result("a", 0.5)], [_result("b", 0.7)]])

        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_hyde_response("hypothetical paragraph"))

        results = await HyDERetrieval().retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        assert reader.retrieve.await_count == 2
        ids = {r.document.id for r in results}
        assert ids == {"a", "b"}

    @pytest.mark.asyncio
    async def test_dedupes_by_id_keeping_highest_score(self):
        reader = MagicMock()
        reader.retrieve = AsyncMock(side_effect=[[_result("d", 0.4)], [_result("d", 0.9)]])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_hyde_response("hyp"))

        results = await HyDERetrieval().retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        assert len(results) == 1
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_no_chat_model_falls_back_to_single_query(self):
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.5)])

        results = await HyDERetrieval().retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=None
        )
        assert reader.retrieve.await_count == 1
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_external_transformers_compose(self):
        """External non-pre_strategy transformers add to the fan-out set."""
        reader = MagicMock()
        # Three queries: original + 1 hyp + 1 from _ExternalTransformer.
        reader.retrieve = AsyncMock(
            side_effect=[
                [_result("a", 0.4)],
                [_result("b", 0.6)],
                [_result("c", 0.8)],
            ]
        )
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_hyde_response("hyp"))

        results = await HyDERetrieval().retrieve(
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
        """Pre_strategy transformers in the list are silently ignored —
        they belong to the agent-entry path, not the strategy."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(side_effect=[[_result("a", 0.5)], [_result("b", 0.7)]])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_hyde_response("hyp"))

        results = await HyDERetrieval().retrieve(
            query="orig",
            namespace="kb",
            scope=_scope(),
            k=5,
            reader=reader,
            chat_model=chat_model,
            transformers=[_PreStrategyDouble()],
        )
        assert reader.retrieve.await_count == 2  # original + 1 hyp; no pre_strategy run
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_hyde_failure_falls_back_to_original_only(self):
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.5)])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        results = await HyDERetrieval().retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        # Only the original query reaches the reader when hyp fails.
        assert reader.retrieve.await_count == 1
        assert len(results) == 1


class TestFromConfig:
    def test_reads_n_hypothetical(self):
        cfg = OrchidRetrievalConfig(hyde=OrchidHydeConfig(n_hypothetical=4))
        strategy = HyDERetrieval.from_config(cfg)
        assert strategy._n_hypothetical == 4

    def test_default_when_no_config(self):
        strategy = HyDERetrieval.from_config(None)
        assert strategy._n_hypothetical == 1

    def test_default_when_config_missing_hyde_block(self):
        # Pass an unrelated object; ``from_config`` must default cleanly.
        strategy = HyDERetrieval.from_config(object())
        assert strategy._n_hypothetical == 1

    @pytest.mark.asyncio
    async def test_n_hypothetical_drives_fan_out(self):
        """With n_hypothetical=3, the LLM is asked once but fans out 3
        hypotheticals → reader called 4× (original + 3 hyps)."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[])
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=_hyde_response("p1\np2\np3"))

        await HyDERetrieval(n_hypothetical=3).retrieve(
            query="orig", namespace="kb", scope=_scope(), k=5, reader=reader, chat_model=chat_model
        )
        assert reader.retrieve.await_count == 4
