"""Tests for ``OrchidRuntime.{doc_store, graph_store, sparse_encoder}``."""

from __future__ import annotations

import pytest

from orchid_ai.core.doc_store import OrchidDocStore
from orchid_ai.core.graph_store import OrchidGraphStore
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.core.sparse import OrchidSparseEncoder, OrchidSparseVector
from orchid_ai.rag.backends.null import NullDocStore, NullGraphStore
from orchid_ai.rag.sparse.bm25 import BM25Encoder
from orchid_ai.runtime import OrchidRuntime


class _DocStub(OrchidDocStore):
    async def put(self, doc_id, content, metadata):
        return None

    async def get(self, doc_id):
        return None

    async def get_many(self, doc_ids):
        return {}


class _GraphStub(OrchidGraphStore):
    async def upsert_entities(self, entities, scope):
        return None

    async def upsert_edges(self, edges, scope):
        return None

    async def find_entities(self, *, query, scope, type_filter=None, k=10):
        return []

    async def neighbours(self, entity_ids, *, scope, max_hops=2, relation_filter=None):
        return ([], [])


class _SparseStub(OrchidSparseEncoder):
    async def encode_documents(self, texts):
        return [OrchidSparseVector(indices=[1], values=[0.5]) for _ in texts]

    async def encode_query(self, text):
        return OrchidSparseVector(indices=[1], values=[0.5])


class TestDefaults:
    def test_doc_store_defaults_to_null(self):
        rt = OrchidRuntime()
        store = rt.get_doc_store()
        assert isinstance(store, NullDocStore)

    def test_graph_store_defaults_to_null(self):
        rt = OrchidRuntime()
        store = rt.get_graph_store()
        assert isinstance(store, NullGraphStore)

    def test_sparse_encoder_defaults_to_bm25(self):
        rt = OrchidRuntime()
        encoder = rt.get_sparse_encoder()
        assert isinstance(encoder, BM25Encoder)


class TestInjection:
    def test_doc_store_returned_when_provided(self):
        stub = _DocStub()
        rt = OrchidRuntime(doc_store=stub)
        assert rt.get_doc_store() is stub

    def test_graph_store_returned_when_provided(self):
        stub = _GraphStub()
        rt = OrchidRuntime(graph_store=stub)
        assert rt.get_graph_store() is stub

    def test_sparse_encoder_returned_when_provided(self):
        stub = _SparseStub()
        rt = OrchidRuntime(sparse_encoder=stub)
        assert rt.get_sparse_encoder() is stub


class TestNullStoreBehaviour:
    @pytest.mark.asyncio
    async def test_null_doc_store_is_silent(self):
        store = NullDocStore()
        await store.put("d1", "hi", {})
        assert await store.get("d1") is None
        assert await store.get_many(["d1", "d2"]) == {}

    @pytest.mark.asyncio
    async def test_null_graph_store_is_silent(self):
        store = NullGraphStore()
        scope = OrchidRAGScope(tenant_id="t")
        await store.upsert_entities([], scope)
        await store.upsert_edges([], scope)
        ents = await store.find_entities(query="x", scope=scope)
        assert ents == []
        ents, edges = await store.neighbours([], scope=scope)
        assert ents == [] and edges == []
