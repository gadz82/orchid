"""ADR-027: ``metadata_filters`` reaches ``reader.retrieve`` through every strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.core.sparse import OrchidSparseEncoder, OrchidSparseVector
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore
from orchid_ai.rag.strategies.graph_rag import GraphRAGRetrieval
from orchid_ai.rag.strategies.hybrid import HybridRetrieval
from orchid_ai.rag.strategies.hyde import HyDERetrieval
from orchid_ai.rag.strategies.multi_query import MultiQueryRetrieval
from orchid_ai.rag.strategies.simple import SimpleRetrieval


_FILTERS = {"status": "published", "language": ["en"]}


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _result(doc_id: str, score: float = 0.5) -> OrchidSearchResult:
    return OrchidSearchResult(
        document=Document(id=doc_id, page_content=doc_id, metadata={}),
        score=score,
    )


class _StaticEncoder(OrchidSparseEncoder):
    """Test double — returns a fixed sparse vector for any text."""

    async def encode_documents(self, texts, namespace=None):
        return [OrchidSparseVector(indices=[1], values=[1.0]) for _ in texts]

    async def encode_query(self, text, namespace=None):
        return OrchidSparseVector(indices=[1], values=[1.0])


def _capturing_reader() -> MagicMock:
    """Mock reader that records every retrieve call's kwargs."""
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[_result("doc-a", 0.5)])
    reader.retrieve_sparse = AsyncMock(return_value=[_result("sparse-a", 0.7)])
    return reader


class TestSimpleRetrieval:
    @pytest.mark.asyncio
    async def test_metadata_filters_propagate(self):
        reader = _capturing_reader()
        await SimpleRetrieval().retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            metadata_filters=_FILTERS,
        )
        kwargs = reader.retrieve.await_args.kwargs
        assert kwargs["metadata_filters"] == _FILTERS


class TestMultiQueryRetrieval:
    @pytest.mark.asyncio
    async def test_metadata_filters_propagate_to_every_fan_out(self):
        reader = _capturing_reader()
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="v1\nv2\nv3"))
        await MultiQueryRetrieval(num_queries=3).retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            chat_model=chat_model,
            metadata_filters=_FILTERS,
        )
        # Original + 3 variations = 4 retrieve calls; every one carries the filters.
        assert reader.retrieve.await_count >= 4
        for call in reader.retrieve.await_args_list:
            assert call.kwargs["metadata_filters"] == _FILTERS


class TestHyDERetrieval:
    @pytest.mark.asyncio
    async def test_metadata_filters_propagate_to_every_fan_out(self):
        reader = _capturing_reader()
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="hypothetical"))
        await HyDERetrieval(n_hypothetical=1).retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            chat_model=chat_model,
            metadata_filters=_FILTERS,
        )
        # Original + 1 hypothetical = 2 retrieve calls; both carry the filters.
        assert reader.retrieve.await_count >= 2
        for call in reader.retrieve.await_args_list:
            assert call.kwargs["metadata_filters"] == _FILTERS


class TestHybridRetrieval:
    @pytest.mark.asyncio
    async def test_metadata_filters_reach_both_lanes(self):
        reader = _capturing_reader()
        await HybridRetrieval(sparse_encoder=_StaticEncoder()).retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            metadata_filters=_FILTERS,
        )
        # Dense lane.
        dense_kwargs = reader.retrieve.await_args.kwargs
        assert dense_kwargs["metadata_filters"] == _FILTERS
        # Sparse lane.
        sparse_kwargs = reader.retrieve_sparse.await_args.kwargs
        assert sparse_kwargs["metadata_filters"] == _FILTERS


class TestGraphRAGRetrieval:
    @pytest.mark.asyncio
    async def test_metadata_filters_reach_vector_chunk_fetch(self):
        reader = _capturing_reader()
        scope = _scope()
        store = InMemoryGraphStore()
        # Seed entities so the strategy doesn't fall back to SimpleRetrieval.
        from orchid_ai.core.graph_store import OrchidEdge, OrchidEntity

        await store.upsert_entities(
            [OrchidEntity(id="A", type="thing", name="A"), OrchidEntity(id="B", type="thing", name="B")],
            scope,
        )
        await store.upsert_edges([OrchidEdge(source_id="A", target_id="B", relation="knows")], scope)

        await GraphRAGRetrieval(max_hops=1, fuse_with_vectors=True).retrieve(
            query="A",
            namespace="kb",
            scope=scope,
            k=3,
            reader=reader,
            graph_store=store,
            metadata_filters=_FILTERS,
        )
        kwargs = reader.retrieve.await_args.kwargs
        assert kwargs["metadata_filters"] == _FILTERS

    @pytest.mark.asyncio
    async def test_metadata_filters_propagate_through_fallback(self):
        """Missing graph store → falls back to SimpleRetrieval which must
        still honour the metadata_filters."""
        reader = _capturing_reader()
        await GraphRAGRetrieval().retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            graph_store=None,
            metadata_filters=_FILTERS,
        )
        kwargs = reader.retrieve.await_args.kwargs
        assert kwargs["metadata_filters"] == _FILTERS


class TestNullFiltersPath:
    """No metadata_filters → ``None`` propagates cleanly."""

    @pytest.mark.asyncio
    async def test_simple_passes_none(self):
        reader = _capturing_reader()
        await SimpleRetrieval().retrieve(query="q", namespace="kb", scope=_scope(), k=3, reader=reader)
        assert reader.retrieve.await_args.kwargs["metadata_filters"] is None
