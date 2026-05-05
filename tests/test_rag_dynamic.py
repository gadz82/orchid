"""Tests for ``orchid_ai.rag.dynamic.inject_to_rag``."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.ingestion import OrchidChunk, OrchidIngestionStrategy
from orchid_ai.core.repository import OrchidVectorReader, OrchidVectorWriter
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.rag.dynamic import inject_to_rag
from orchid_ai.rag.scopes import OrchidRAGScope


class _FakeReaderWriter(OrchidVectorReader, OrchidVectorWriter):
    """In-memory store that records every ``upsert`` call."""

    def __init__(self) -> None:
        self.upserted: list[tuple] = []

    async def retrieve(self, query, namespace, k=5, scope=None, metadata_filters=None):
        return []

    async def index(self, documents, namespace):
        pass

    async def upsert(self, documents, namespace):
        self.upserted.append((documents, namespace))

    async def delete(self, document_ids, namespace):
        pass


class _RecordingIngestion(OrchidIngestionStrategy):
    """Strategy double that records its inputs and returns one chunk per line."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ingest(self, *, text, filename, scope, doc_store=None, embeddings=None):
        self.calls.append({"text": text, "filename": filename, "scope": scope})
        return [
            OrchidChunk(
                text=line,
                metadata={
                    "tenant_id": scope.tenant_id,
                    "user_id": scope.user_id,
                    "chat_id": scope.chat_id,
                    "scope": "chat_shared",
                    "source_file": filename,
                    "chunk_index": idx,
                },
            )
            for idx, line in enumerate(text.splitlines())
            if line.strip()
        ]


@pytest.fixture
def scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1")


class TestInjectToRag:
    @pytest.mark.asyncio
    async def test_null_store_no_ops(self, scope):
        count = await inject_to_rag(
            NullVectorReader(),
            tool_name="search",
            tool_result="hello",
            namespace="ns",
            scope=scope,
            ingestion=_RecordingIngestion(),
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_error_dict_skipped(self, scope):
        rw = _FakeReaderWriter()
        ingestion = _RecordingIngestion()
        count = await inject_to_rag(
            rw,
            tool_name="search",
            tool_result={"error": "boom"},
            namespace="ns",
            scope=scope,
            ingestion=ingestion,
        )
        assert count == 0
        assert rw.upserted == []
        assert ingestion.calls == []  # strategy is never invoked for error results

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self, scope):
        rw = _FakeReaderWriter()
        count = await inject_to_rag(
            rw,
            tool_name="search",
            tool_result="",
            namespace="ns",
            scope=scope,
            ingestion=_RecordingIngestion(),
        )
        assert count == 0
        assert rw.upserted == []

    @pytest.mark.asyncio
    async def test_string_result_flows_through_strategy(self, scope):
        rw = _FakeReaderWriter()
        ingestion = _RecordingIngestion()
        count = await inject_to_rag(
            rw,
            tool_name="search_kb",
            tool_result="line one\nline two\nline three",
            namespace="kb",
            scope=scope,
            ingestion=ingestion,
        )
        assert count == 3
        assert len(rw.upserted) == 1
        docs, ns = rw.upserted[0]
        assert ns == "kb"
        assert len(docs) == 3
        # Strategy received the original text + a tool-prefixed filename + the scope.
        assert ingestion.calls[0]["text"] == "line one\nline two\nline three"
        assert ingestion.calls[0]["filename"] == "tool:search_kb"
        assert ingestion.calls[0]["scope"] is scope

    @pytest.mark.asyncio
    async def test_dict_result_is_json_serialised(self, scope):
        rw = _FakeReaderWriter()
        ingestion = _RecordingIngestion()
        await inject_to_rag(
            rw,
            tool_name="lookup",
            tool_result={"name": "Acme", "count": 42},
            namespace="ns",
            scope=scope,
            ingestion=ingestion,
        )
        passed_text = ingestion.calls[0]["text"]
        assert "Acme" in passed_text
        assert "42" in passed_text

    @pytest.mark.asyncio
    async def test_metadata_carries_dynamic_markers(self, scope):
        rw = _FakeReaderWriter()
        await inject_to_rag(
            rw,
            tool_name="search_kb",
            tool_result="alpha\nbeta",
            namespace="kb",
            scope=scope,
            ingestion=_RecordingIngestion(),
        )
        docs, _ = rw.upserted[0]
        meta = docs[0].metadata
        # Strategy-provided scope fields preserved.
        assert meta["tenant_id"] == "t-1"
        assert meta["chat_id"] == "c-1"
        assert meta["scope"] == "chat_shared"
        # Dynamic-injection markers added on top.
        assert meta["source_tool"] == "search_kb"
        assert meta["dynamic"] is True
        assert isinstance(meta["injected_at"], float)

    @pytest.mark.asyncio
    async def test_no_chunks_no_upsert(self, scope):
        rw = _FakeReaderWriter()

        class EmptyStrategy(OrchidIngestionStrategy):
            async def ingest(self, *, text, filename, scope, doc_store=None, embeddings=None):
                return []

        count = await inject_to_rag(
            rw,
            tool_name="search",
            tool_result="some text",
            namespace="ns",
            scope=scope,
            ingestion=EmptyStrategy(),
        )
        assert count == 0
        assert rw.upserted == []

    @pytest.mark.asyncio
    async def test_upsert_failure_returns_zero_and_does_not_raise(self, scope):
        class FlakyStore(_FakeReaderWriter):
            async def upsert(self, documents, namespace):
                raise RuntimeError("disk full")

        count = await inject_to_rag(
            FlakyStore(),
            tool_name="search",
            tool_result="content",
            namespace="ns",
            scope=scope,
            ingestion=_RecordingIngestion(),
        )
        assert count == 0
