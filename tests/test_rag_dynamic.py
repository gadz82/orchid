"""Tests for src.rag.dynamic — inject_to_rag + _tool_data_to_documents."""

from __future__ import annotations

import json

import pytest

from orchid_ai.core.repository import Document, VectorReader, VectorWriter
from orchid_ai.rag.dynamic import _tool_data_to_documents, inject_to_rag
from orchid_ai.rag.null import NullVectorReader
from orchid_ai.rag.scopes import RAGScope


# ── Mock that implements both VectorReader and VectorWriter ─────


class MockReaderWriter(VectorReader, VectorWriter):
    def __init__(self):
        self.upserted: list[tuple] = []

    async def retrieve(self, query, namespace, k=5, scope=None):
        return []

    async def index(self, documents, namespace):
        pass

    async def upsert(self, documents, namespace):
        self.upserted.append((documents, namespace))

    async def delete(self, document_ids, namespace):
        pass


# ── inject_to_rag ──────────────────────────────────────────────


class TestInjectToRag:
    @pytest.fixture
    def scope(self):
        return RAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1")

    @pytest.mark.asyncio
    async def test_null_reader_returns_zero(self, scope):
        reader = NullVectorReader()
        count = await inject_to_rag(reader, mcp_data={"key": "value"}, namespace="ns", scope=scope)
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_mcp_data_returns_zero(self, scope):
        rw = MockReaderWriter()
        count = await inject_to_rag(rw, mcp_data={}, namespace="ns", scope=scope)
        assert count == 0

    @pytest.mark.asyncio
    async def test_error_key_returns_zero(self, scope):
        rw = MockReaderWriter()
        count = await inject_to_rag(rw, mcp_data={"error": "something went wrong"}, namespace="ns", scope=scope)
        assert count == 0

    @pytest.mark.asyncio
    async def test_reader_writer_upserts_and_returns_count(self, scope):
        rw = MockReaderWriter()
        data = {"courses": "Python 101", "enrollments": "5 users"}
        count = await inject_to_rag(rw, mcp_data=data, namespace="learning", scope=scope)
        assert count == 2
        assert len(rw.upserted) == 1
        docs, ns = rw.upserted[0]
        assert ns == "learning"
        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_data_with_error_key_mixed_still_skips_error(self, scope):
        """If data has 'error' as one of several keys, inject_to_rag returns 0
        because the top-level check catches it."""
        rw = MockReaderWriter()
        data = {"error": "oops", "courses": "data"}
        # The function checks `"error" in mcp_data` at the top level
        count = await inject_to_rag(rw, mcp_data=data, namespace="ns", scope=scope)
        assert count == 0


# ── _tool_data_to_documents ────────────────────────────────────


class TestToolDataToDocuments:
    @pytest.fixture
    def scope(self):
        return RAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1")

    def test_creates_one_document_per_key(self, scope):
        data = {"a": "text_a", "b": "text_b", "c": "text_c"}
        docs = _tool_data_to_documents(data, scope, source_tool="test")
        assert len(docs) == 3

    def test_skips_error_keys(self, scope):
        data = {"error": "bad", "good": "data"}
        docs = _tool_data_to_documents(data, scope, source_tool="test")
        assert len(docs) == 1
        assert docs[0].page_content == "data"

    def test_deterministic_ids(self, scope):
        data = {"key": "value"}
        docs1 = _tool_data_to_documents(data, scope, source_tool="t1")
        docs2 = _tool_data_to_documents(data, scope, source_tool="t2")
        # Same data + same scope → same document ID (source_tool not in the ID hash)
        assert docs1[0].id == docs2[0].id

    def test_metadata_includes_scope_fields(self, scope):
        data = {"info": "hello"}
        docs = _tool_data_to_documents(data, scope, source_tool="my_tool")
        meta = docs[0].metadata
        assert meta["tenant_id"] == "t-1"
        assert meta["user_id"] == "u-1"
        assert meta["chat_id"] == "c-1"
        assert meta["scope"] == "chat_shared"
        assert meta["source_tool"] == "my_tool"
        assert meta["dynamic"] is True
        assert "injected_at" in meta
        assert isinstance(meta["injected_at"], float)

    def test_large_text_truncated(self, scope):
        long_text = "x" * 3000
        data = {"big": long_text}
        docs = _tool_data_to_documents(data, scope, source_tool="test")
        assert len(docs[0].page_content) < 3000
        assert docs[0].page_content.endswith("... [truncated]")
        # Truncated at 2000 chars + the suffix
        assert docs[0].page_content[:2000] == "x" * 2000

    def test_non_string_values_json_serialized(self, scope):
        data = {"nums": [1, 2, 3], "obj": {"nested": True}}
        docs = _tool_data_to_documents(data, scope, source_tool="test")
        assert len(docs) == 2
        # Find the document for "nums"
        nums_doc = next(d for d in docs if "nums" in d.id)
        parsed = json.loads(nums_doc.page_content)
        assert parsed == [1, 2, 3]

    def test_document_type_is_correct(self, scope):
        data = {"k": "v"}
        docs = _tool_data_to_documents(data, scope, source_tool="test")
        assert isinstance(docs[0], Document)
