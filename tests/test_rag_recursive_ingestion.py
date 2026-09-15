"""Tests for ``RecursiveIngestion``."""

from __future__ import annotations

import pytest

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.chunker import ChunkConfig
from orchid_ai.documents.strategies.recursive import RecursiveIngestion


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")


class TestFlatChunking:
    @pytest.mark.asyncio
    async def test_returns_one_chunk_for_small_text(self):
        chunks = await RecursiveIngestion().ingest(text="Short paragraph.", filename="x.txt", scope=_scope())
        assert len(chunks) == 1
        assert chunks[0].text == "Short paragraph."

    @pytest.mark.asyncio
    async def test_splits_long_text(self):
        cfg = ChunkConfig(chunk_size=200, chunk_overlap=50)
        text = "Word " * 500  # ~2500 chars
        chunks = await RecursiveIngestion(cfg).ingest(text=text, filename="big.txt", scope=_scope())
        assert len(chunks) > 1
        for c in chunks:
            assert "parent_content" not in c.metadata
            assert c.metadata["source_file"] == "big.txt"
            assert c.metadata["scope"] == "chat_shared"

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        assert await RecursiveIngestion().ingest(text="", filename="x", scope=_scope()) == []
        assert await RecursiveIngestion().ingest(text="   \n\t", filename="x", scope=_scope()) == []

    @pytest.mark.asyncio
    async def test_metadata_carries_scope_fields(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")
        chunks = await RecursiveIngestion().ingest(text="hello", filename="x.txt", scope=scope)
        meta = chunks[0].metadata
        assert meta["tenant_id"] == "t1"
        assert meta["user_id"] == "u1"
        assert meta["chat_id"] == "c1"
        assert meta["source_file"] == "x.txt"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 1
        assert meta["chunk_id"]


class TestScopeLevelMetadata:
    @pytest.mark.asyncio
    async def test_tenant_scope_level(self):
        chunks = await RecursiveIngestion().ingest(text="hello", filename="x.txt", scope=OrchidRAGScope(tenant_id="t1"))
        assert chunks[0].metadata["scope"] == "tenant"

    @pytest.mark.asyncio
    async def test_user_scope_level(self):
        chunks = await RecursiveIngestion().ingest(
            text="hello", filename="x.txt", scope=OrchidRAGScope(tenant_id="t1", user_id="u1")
        )
        assert chunks[0].metadata["scope"] == "user"

    @pytest.mark.asyncio
    async def test_chat_shared_scope_level(self):
        chunks = await RecursiveIngestion().ingest(
            text="hello", filename="x.txt", scope=OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")
        )
        assert chunks[0].metadata["scope"] == "chat_shared"

    @pytest.mark.asyncio
    async def test_chat_agent_scope_level(self):
        chunks = await RecursiveIngestion().ingest(
            text="hello",
            filename="x.txt",
            scope=OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1"),
        )
        assert chunks[0].metadata["scope"] == "chat_agent"


class TestParentChildChunking:
    @pytest.mark.asyncio
    async def test_parent_in_metadata_when_parent_size_set(self):
        cfg = ChunkConfig(chunk_size=200, parent_chunk_size=600, parent_chunk_overlap=50)
        text = "Word " * 500
        chunks = await RecursiveIngestion(cfg).ingest(text=text, filename="t.txt", scope=_scope())
        assert chunks
        for c in chunks:
            assert "parent_content" in c.metadata
            assert "parent_index" in c.metadata
            assert len(c.text) <= 220
