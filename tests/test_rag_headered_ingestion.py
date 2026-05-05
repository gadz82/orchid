"""Tests for ``HeaderedIngestion`` and ``ContextualHeaderPostProcessor``."""

from __future__ import annotations

import pytest

from orchid_ai.core.ingestion import OrchidChunk
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.chunker import ChunkConfig
from orchid_ai.documents.post_processors.contextual_headers import (
    ContextualHeaderPostProcessor,
)
from orchid_ai.documents.strategies.headered import HeaderedIngestion


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")


class TestHeaderedIngestion:
    @pytest.mark.asyncio
    async def test_chunk_text_starts_with_title_and_section(self):
        text = "# Introduction\nFirst paragraph.\n\n## Body\nSecond paragraph."
        chunks = await HeaderedIngestion(ChunkConfig(chunk_size=400)).ingest(
            text=text, filename="my_document.md", scope=_scope()
        )
        assert chunks
        for c in chunks:
            # Required prefix:
            #   ^# {title}\n## {section}\n
            assert c.text.startswith("# My Document\n## ")

    @pytest.mark.asyncio
    async def test_section_metadata_set(self):
        text = "# Introduction\nFirst paragraph.\n\n## Body\nSecond paragraph."
        chunks = await HeaderedIngestion(ChunkConfig(chunk_size=400)).ingest(
            text=text, filename="my_doc.md", scope=_scope()
        )
        for c in chunks:
            assert c.metadata["title"] == "My Doc"
            assert c.metadata["section"]
            assert c.metadata["contextual_header"] is True


class TestContextualHeaderPostProcessor:
    @pytest.mark.asyncio
    async def test_uses_nearest_preceding_heading(self):
        text = (
            "# Doc Title\n\nIntro paragraph.\n\n"
            "## Section One\n\nSection one body.\n\n"
            "## Section Two\n\nSection two body.\n"
        )
        chunks = [
            OrchidChunk(text="Section one body.", metadata={"chunk_index": 0}),
            OrchidChunk(text="Section two body.", metadata={"chunk_index": 1}),
        ]
        out = await ContextualHeaderPostProcessor().process(chunks, text=text, filename="x.md")
        assert out[0].metadata["section"] == "Section One"
        assert out[1].metadata["section"] == "Section Two"

    @pytest.mark.asyncio
    async def test_default_section_when_no_heading(self):
        text = "Just a flat document with no headings whatsoever."
        chunk = OrchidChunk(text="Just a flat document.", metadata={})
        out = await ContextualHeaderPostProcessor().process([chunk], text=text, filename="flat.txt")
        assert out[0].metadata["section"] == "Document"
        assert out[0].text.startswith("# Flat\n## Document\n")

    @pytest.mark.asyncio
    async def test_idempotent_when_already_processed(self):
        chunk = OrchidChunk(text="hello", metadata={"contextual_header": True})
        out = await ContextualHeaderPostProcessor().process([chunk], text="hello", filename="x.md")
        assert out == [chunk]

    @pytest.mark.asyncio
    async def test_empty_chunk_list(self):
        assert await ContextualHeaderPostProcessor().process([], text="x", filename="x.md") == []

    def test_filename_to_title_handles_underscores(self):
        from orchid_ai.documents.post_processors.contextual_headers import _filename_to_title

        assert _filename_to_title("my_great_doc.txt") == "My Great Doc"
        assert _filename_to_title("hyphen-name.md") == "Hyphen Name"
        assert _filename_to_title("") == "Document"
