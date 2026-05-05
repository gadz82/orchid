"""Tests for ``HierarchicalIngestion``."""

from __future__ import annotations

import pytest

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.chunker import ChunkConfig
from orchid_ai.documents.strategies.hierarchical import HierarchicalIngestion
from orchid_ai.rag.backends.in_memory_doc_store import InMemoryDocStore
from orchid_ai.rag.backends.null import NullDocStore


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")


_CFG = ChunkConfig(chunk_size=200, parent_chunk_size=600, parent_chunk_overlap=50)


class TestWithRealDocStore:
    @pytest.mark.asyncio
    async def test_writes_parents_to_doc_store(self):
        store = InMemoryDocStore()
        text = "Word " * 600
        chunks = await HierarchicalIngestion(_CFG).ingest(text=text, filename="t.txt", scope=_scope(), doc_store=store)
        assert chunks

        # Every child chunk carries a parent_id, and that parent_id is
        # resolvable via the docstore.
        parent_ids = sorted({c.metadata["parent_id"] for c in chunks})
        records = await store.get_many(parent_ids)
        assert set(records) == set(parent_ids), (
            f"Some parent_ids missing from docstore: {set(parent_ids) - set(records)}"
        )

    @pytest.mark.asyncio
    async def test_no_parent_content_in_metadata_with_real_store(self):
        store = InMemoryDocStore()
        chunks = await HierarchicalIngestion(_CFG).ingest(
            text="Word " * 600, filename="t.txt", scope=_scope(), doc_store=store
        )
        for c in chunks:
            assert "parent_content" not in c.metadata, (
                "Real docstore should not duplicate parent text into chunk metadata"
            )

    @pytest.mark.asyncio
    async def test_parent_content_round_trips(self):
        store = InMemoryDocStore()
        text = "First paragraph alpha alpha alpha.\n\n" + ("Beta words " * 100)
        chunks = await HierarchicalIngestion(_CFG).ingest(text=text, filename="t.txt", scope=_scope(), doc_store=store)

        # Pick one child, look up its parent in the docstore — content
        # should be a non-empty string and contain the child's text
        # (since the child is a sub-string of the parent).
        sample = chunks[0]
        record = await store.get(sample.metadata["parent_id"])
        assert record is not None
        parent_content, _meta = record
        assert parent_content
        assert sample.text.split()[0] in parent_content


class TestWithNullDocStore:
    @pytest.mark.asyncio
    async def test_falls_back_to_parent_in_metadata(self):
        chunks = await HierarchicalIngestion(_CFG).ingest(
            text="Word " * 600, filename="t.txt", scope=_scope(), doc_store=NullDocStore()
        )
        assert chunks
        for c in chunks:
            assert "parent_content" in c.metadata, "NullDocStore should trigger the parent-in-metadata fallback"
            assert c.metadata["parent_content"]

    @pytest.mark.asyncio
    async def test_falls_back_when_doc_store_is_none(self):
        chunks = await HierarchicalIngestion(_CFG).ingest(
            text="Word " * 600, filename="t.txt", scope=_scope(), doc_store=None
        )
        assert chunks
        for c in chunks:
            assert "parent_content" in c.metadata


class TestMetadataInvariants:
    @pytest.mark.asyncio
    async def test_chunk_metadata_carries_scope(self):
        store = InMemoryDocStore()
        chunks = await HierarchicalIngestion(_CFG).ingest(
            text="Word " * 300, filename="x.txt", scope=_scope(), doc_store=store
        )
        for c in chunks:
            assert c.metadata["tenant_id"] == "t1"
            assert c.metadata["user_id"] == "u1"
            assert c.metadata["chat_id"] == "c1"
            assert c.metadata["scope"] == "chat_shared"
            assert c.metadata["source_file"] == "x.txt"
            assert c.metadata["ingestion_strategy"] == "hierarchical"
            assert c.metadata["parent_id"]
            assert c.metadata["chunk_id"].startswith(c.metadata["parent_id"])

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        store = InMemoryDocStore()
        assert await HierarchicalIngestion().ingest(text="", filename="x.txt", scope=_scope(), doc_store=store) == []

    @pytest.mark.asyncio
    async def test_parent_size_auto_derives_when_zero(self):
        """ChunkConfig with parent_chunk_size=0 (default) still produces parents."""
        store = InMemoryDocStore()
        chunks = await HierarchicalIngestion(ChunkConfig(chunk_size=200)).ingest(
            text="Word " * 600, filename="t.txt", scope=_scope(), doc_store=store
        )
        assert chunks
        # Auto-derived parent size = chunk_size * 4 = 800; parents
        # written to docstore should exist.
        parent_ids = {c.metadata["parent_id"] for c in chunks}
        records = await store.get_many(list(parent_ids))
        assert records
