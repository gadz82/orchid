"""Tests for the four backend registries."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.doc_store import OrchidDocStore
from orchid_ai.core.graph_store import OrchidGraphStore
from orchid_ai.core.repository import OrchidVectorReader
from orchid_ai.core.sparse import OrchidSparseEncoder, OrchidSparseVector
from orchid_ai.rag.backends.null import NullDocStore, NullGraphStore, NullVectorReader
from orchid_ai.rag.factory import (
    DOC_STORE_BACKEND_REGISTRY,
    GRAPH_STORE_BACKEND_REGISTRY,
    VECTOR_BACKEND_REGISTRY,
    build_doc_store,
    build_graph_store,
    build_reader,
    build_sparse_encoder,
    register_doc_store_backend,
    register_graph_store_backend,
    register_sparse_encoder_backend,
    register_vector_backend,
)
from orchid_ai.rag.sparse import SPARSE_ENCODER_REGISTRY, BM25Encoder


# ── Vector backend ────────────────────────────────────────────


class TestVectorBackendRegistry:
    def test_null_built_in(self):
        assert "null" in VECTOR_BACKEND_REGISTRY
        assert isinstance(build_reader(vector_backend="null"), NullVectorReader)

    def test_qdrant_not_built_in(self):
        # Qdrant moved to orchid-rag-qdrant plugin package.
        assert "qdrant" not in VECTOR_BACKEND_REGISTRY

    def test_register_custom_and_build(self):
        class _MyReader(OrchidVectorReader):
            async def retrieve(self, query, namespace, k=5, scope=None):
                return []

        def _builder(**_settings: Any) -> OrchidVectorReader:
            return _MyReader()

        register_vector_backend("custom-vec", _builder)
        try:
            reader = build_reader(vector_backend="custom-vec")
            assert isinstance(reader, _MyReader)
        finally:
            VECTOR_BACKEND_REGISTRY.pop("custom-vec", None)

    def test_unknown_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="pip install"):
            build_reader(vector_backend="qdrant")

    def test_overwrite_logs_warning(self, caplog):
        with caplog.at_level("WARNING"):

            def _other(**_):
                return NullVectorReader()

            register_vector_backend("null", _other)
        # Restore the default builder so other tests stay green.
        from orchid_ai.rag.factory import _build_null_reader

        register_vector_backend("null", _build_null_reader)
        assert any("null" in rec.message for rec in caplog.records)


# ── Doc store backend ─────────────────────────────────────────


class TestDocStoreBackendRegistry:
    def test_null_built_in(self):
        assert "null" in DOC_STORE_BACKEND_REGISTRY
        assert isinstance(build_doc_store(doc_store_backend="null"), NullDocStore)

    def test_in_memory_built_in(self):
        from orchid_ai.rag.backends.in_memory_doc_store import InMemoryDocStore

        assert "in_memory" in DOC_STORE_BACKEND_REGISTRY
        assert isinstance(build_doc_store(doc_store_backend="in_memory"), InMemoryDocStore)

    def test_qdrant_not_registered(self):
        # Qdrant moved to orchid-rag-qdrant plugin package.
        assert "qdrant" not in DOC_STORE_BACKEND_REGISTRY

    def test_register_custom(self):
        class _MyStore(OrchidDocStore):
            async def put(self, doc_id, content, metadata):
                return None

            async def get(self, doc_id):
                return None

            async def get_many(self, doc_ids):
                return {}

        register_doc_store_backend("custom-doc", lambda **_: _MyStore())
        try:
            assert isinstance(build_doc_store(doc_store_backend="custom-doc"), _MyStore)
        finally:
            DOC_STORE_BACKEND_REGISTRY.pop("custom-doc", None)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="register_doc_store_backend"):
            build_doc_store(doc_store_backend="missing")


# ── Graph store backend ───────────────────────────────────────


class TestGraphStoreBackendRegistry:
    def test_null_built_in(self):
        assert "null" in GRAPH_STORE_BACKEND_REGISTRY
        assert isinstance(build_graph_store(graph_store_backend="null"), NullGraphStore)

    def test_in_memory_built_in(self):
        from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore

        assert "in_memory" in GRAPH_STORE_BACKEND_REGISTRY
        assert isinstance(build_graph_store(graph_store_backend="in_memory"), InMemoryGraphStore)

    def test_neo4j_not_registered(self):
        # Neo4j moved to orchid-rag-neo4j plugin package.
        assert "neo4j" not in GRAPH_STORE_BACKEND_REGISTRY

    def test_register_custom(self):
        class _MyGraph(OrchidGraphStore):
            async def upsert_entities(self, entities, scope):
                return None

            async def upsert_edges(self, edges, scope):
                return None

            async def find_entities(self, *, query, scope, type_filter=None, k=10):
                return []

            async def neighbours(self, entity_ids, *, scope, max_hops=2, relation_filter=None):
                return ([], [])

        register_graph_store_backend("custom-graph", lambda **_: _MyGraph())
        try:
            assert isinstance(build_graph_store(graph_store_backend="custom-graph"), _MyGraph)
        finally:
            GRAPH_STORE_BACKEND_REGISTRY.pop("custom-graph", None)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="register_graph_store_backend"):
            build_graph_store(graph_store_backend="missing")


# ── Sparse encoder backend ────────────────────────────────────


class TestSparseEncoderRegistry:
    def test_bm25_built_in(self):
        assert "bm25" in SPARSE_ENCODER_REGISTRY
        assert isinstance(build_sparse_encoder(sparse_encoder="bm25"), BM25Encoder)

    def test_register_custom(self):
        class _Empty(OrchidSparseEncoder):
            async def encode_documents(self, texts):
                return [OrchidSparseVector(indices=[], values=[]) for _ in texts]

            async def encode_query(self, text):
                return OrchidSparseVector(indices=[], values=[])

        register_sparse_encoder_backend("empty-sparse", _Empty)
        try:
            assert isinstance(build_sparse_encoder(sparse_encoder="empty-sparse"), _Empty)
        finally:
            SPARSE_ENCODER_REGISTRY.pop("empty-sparse", None)

    def test_unknown_falls_back_to_bm25(self):
        # Sparse encoders fall back rather than raise (every Hybrid setup
        # is expected to use either bm25 or a registered custom encoder).
        encoder = build_sparse_encoder(sparse_encoder="missing")
        assert isinstance(encoder, BM25Encoder)
