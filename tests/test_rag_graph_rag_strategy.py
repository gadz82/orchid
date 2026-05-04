"""Tests for ``GraphRAGRetrieval`` (ADR-026)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.config.schema import OrchidRetrievalConfig
from orchid_ai.config.schema_rag import OrchidGraphRetrievalConfig
from orchid_ai.core.graph_store import OrchidEdge, OrchidEntity
from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore
from orchid_ai.rag.backends.null import NullGraphStore
from orchid_ai.rag.strategies.graph_rag import GraphRAGRetrieval, _default_serialise


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _result(doc_id: str, score: float = 0.5) -> OrchidSearchResult:
    return OrchidSearchResult(
        document=Document(id=doc_id, page_content=doc_id, metadata={}),
        score=score,
    )


def _entity(eid: str, name: str | None = None, type_: str = "thing") -> OrchidEntity:
    return OrchidEntity(id=eid, type=type_, name=name or eid, properties={}, metadata={})


def _edge(src: str, tgt: str, rel: str = "knows") -> OrchidEdge:
    return OrchidEdge(source_id=src, target_id=tgt, relation=rel, properties={}, metadata={})


async def _populate_chain(scope: OrchidRAGScope, depth: int = 3) -> InMemoryGraphStore:
    """Build A → B → Z graph (or A → B → C → Z for depth=3)."""
    store = InMemoryGraphStore()
    if depth == 2:
        await store.upsert_entities([_entity("A"), _entity("B"), _entity("Z")], scope)
        await store.upsert_edges([_edge("A", "B", "supplies"), _edge("B", "Z", "owns")], scope)
    else:
        await store.upsert_entities([_entity("A"), _entity("B"), _entity("C"), _entity("Z")], scope)
        await store.upsert_edges(
            [
                _edge("A", "B", "supplies"),
                _edge("B", "C", "supplies"),
                _edge("C", "Z", "owns"),
            ],
            scope,
        )
    return store


class TestMissingGraphStore:
    @pytest.mark.asyncio
    async def test_no_graph_store_falls_back_to_simple(self, caplog):
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.9)])
        with caplog.at_level("WARNING"):
            results = await GraphRAGRetrieval().retrieve(
                query="anything",
                namespace="kb",
                scope=_scope(),
                k=3,
                reader=reader,
                graph_store=None,
            )
        assert [r.document.id for r in results] == ["a"]
        assert any("graph_store" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_null_graph_store_falls_back_to_simple(self, caplog):
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.9)])
        with caplog.at_level("WARNING"):
            results = await GraphRAGRetrieval().retrieve(
                query="anything",
                namespace="kb",
                scope=_scope(),
                k=3,
                reader=reader,
                graph_store=NullGraphStore(),
            )
        assert [r.document.id for r in results] == ["a"]
        assert any("graph_store" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_no_seed_entities_falls_back_to_simple(self):
        """When the query mentions no known entities, vector-only result."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("v", 0.9)])
        store = InMemoryGraphStore()  # empty
        results = await GraphRAGRetrieval().retrieve(
            query="unknown",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
            graph_store=store,
        )
        assert [r.document.id for r in results] == ["v"]


class TestMultiHop:
    @pytest.mark.asyncio
    async def test_two_hops_reveals_z(self):
        """ADR-026 §Tests: query mentions A; 2-hop walk reveals Z."""
        scope = _scope()
        store = await _populate_chain(scope, depth=2)

        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[])

        results = await GraphRAGRetrieval(max_hops=2, fuse_with_vectors=False).retrieve(
            query="A",
            namespace="kb",
            scope=scope,
            k=5,
            reader=reader,
            graph_store=store,
        )
        assert len(results) == 1
        graph_doc = results[0]
        assert graph_doc.document.metadata["source"] == "graph_rag"
        # Z must appear in the serialised sub-graph.
        assert "Z" in graph_doc.document.page_content
        # And in mentioned_entities metadata.
        assert "Z" in set(graph_doc.document.metadata["mentioned_entities"])

    @pytest.mark.asyncio
    async def test_max_hops_limits_traversal(self):
        """With max_hops=1, Z is too deep and shouldn't be reachable."""
        scope = _scope()
        store = await _populate_chain(scope, depth=3)  # A → B → C → Z

        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[])

        results = await GraphRAGRetrieval(max_hops=1, fuse_with_vectors=False).retrieve(
            query="A",
            namespace="kb",
            scope=scope,
            k=5,
            reader=reader,
            graph_store=store,
        )
        graph_doc = results[0]
        # B reachable in 1 hop; Z is 3 hops away.
        assert "B" in graph_doc.document.page_content
        assert "Z" not in graph_doc.document.page_content


class TestFuseWithVectors:
    @pytest.mark.asyncio
    async def test_false_returns_only_graph_context(self):
        scope = _scope()
        store = await _populate_chain(scope, depth=2)

        reader = MagicMock()
        # Vector lane has hits, but we should NOT see them in the result.
        reader.retrieve = AsyncMock(return_value=[_result("vec-1", 0.9), _result("vec-2", 0.8)])

        results = await GraphRAGRetrieval(max_hops=2, fuse_with_vectors=False).retrieve(
            query="A",
            namespace="kb",
            scope=scope,
            k=5,
            reader=reader,
            graph_store=store,
        )
        assert len(results) == 1
        assert results[0].document.metadata["source"] == "graph_rag"

    @pytest.mark.asyncio
    async def test_true_prepends_graph_context_then_vectors(self):
        scope = _scope()
        store = await _populate_chain(scope, depth=2)

        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("vec-1", 0.9), _result("vec-2", 0.8)])

        results = await GraphRAGRetrieval(max_hops=2, fuse_with_vectors=True).retrieve(
            query="A",
            namespace="kb",
            scope=scope,
            k=3,
            reader=reader,
            graph_store=store,
        )
        # Graph doc first, then vector hits up to k.
        assert results[0].document.metadata["source"] == "graph_rag"
        ids = [r.document.id for r in results[1:]]
        assert "vec-1" in ids


class TestSerialise:
    def test_default_format(self):
        entities = [_entity("supplier:acme", name="ACME"), _entity("product:widget", name="Widget")]
        edges = [_edge("supplier:acme", "product:widget", "supplies")]
        out = _default_serialise(entities, edges)
        assert "[Knowledge graph context]" in out
        assert "ACME" in out
        assert "Widget" in out
        assert "[supplies]" in out

    def test_empty_returns_marker(self):
        assert _default_serialise([], []) == "[Knowledge graph context — empty]"


class TestFromConfig:
    def test_reads_graph_block(self):
        cfg = OrchidRetrievalConfig(
            graph=OrchidGraphRetrievalConfig(max_hops=4, fuse_with_vectors=False, relation_filter=["reports_to"])
        )
        strategy = GraphRAGRetrieval.from_config(cfg)
        assert strategy._max_hops == 4
        assert strategy._fuse_with_vectors is False
        assert strategy._relation_filter == ["reports_to"]

    def test_default_when_no_config(self):
        strategy = GraphRAGRetrieval.from_config(None)
        assert strategy._max_hops == 2
        assert strategy._fuse_with_vectors is True
        assert strategy._relation_filter is None


class TestValidation:
    def test_negative_max_hops_raises(self):
        with pytest.raises(ValueError, match="max_hops"):
            GraphRAGRetrieval(max_hops=-1)

    def test_zero_seed_k_raises(self):
        with pytest.raises(ValueError, match="seed_k"):
            GraphRAGRetrieval(seed_k=0)
