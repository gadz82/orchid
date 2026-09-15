from __future__ import annotations

import pytest

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.strategies import FrontMatterIngestion, get_ingestion_strategy


@pytest.fixture
def scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


@pytest.mark.asyncio
async def test_front_matter_parses_and_injects_metadata(scope: OrchidRAGScope):
    text = """---
page_id: "12345"
title: Notifications
---
# Notifications

This is the body.
"""
    strategy = FrontMatterIngestion(id_field="page_id")
    chunks = await strategy.ingest(text=text, filename="notes.md", scope=scope)

    assert len(chunks) > 0
    meta = chunks[0].metadata
    assert meta["frontmatter_page_id"] == "12345"
    assert meta["frontmatter_title"] == "Notifications"
    assert meta["source_file"] == "notes.md"
    assert "---" not in chunks[0].text  # front-matter delimiters stripped
    assert "This is the body." in chunks[0].text


@pytest.mark.asyncio
async def test_front_matter_id_field_produces_stable_chunk_ids(scope: OrchidRAGScope):
    text = """---
page_id: "12345"
---
Body content here.
"""
    strategy = FrontMatterIngestion(id_field="page_id")
    chunks = await strategy.ingest(text=text, filename="notes.md", scope=scope)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == f"src-12345-{i}"
        assert chunk.metadata["source_id"] == "12345"


@pytest.mark.asyncio
async def test_front_matter_missing_id_field_falls_back(scope: OrchidRAGScope):
    text = """---
title: Notifications
---
Body content here.
"""
    strategy = FrontMatterIngestion(id_field="page_id")
    chunks = await strategy.ingest(text=text, filename="notes.md", scope=scope)

    assert len(chunks) > 0
    # chunk_id should come from inner strategy, not src-...
    assert not chunks[0].metadata["chunk_id"].startswith("src-")


@pytest.mark.asyncio
async def test_front_matter_no_frontmatter_delegates(scope: OrchidRAGScope):
    text = "# No front matter\n\nJust body."
    strategy = FrontMatterIngestion(id_field="page_id")
    chunks = await strategy.ingest(text=text, filename="plain.md", scope=scope)

    assert len(chunks) > 0
    assert "# No front matter" in chunks[0].text
    assert "frontmatter_page_id" not in chunks[0].metadata


@pytest.mark.asyncio
async def test_front_matter_numeric_id_field(scope: OrchidRAGScope):
    text = """---
article_id: 987
---
Body.
"""
    strategy = FrontMatterIngestion(id_field="article_id")
    chunks = await strategy.ingest(text=text, filename="kb.md", scope=scope)

    assert chunks[0].metadata["chunk_id"] == "src-987-0"


@pytest.mark.asyncio
async def test_front_matter_empty_chunks_for_empty_body(scope: OrchidRAGScope):
    text = """---
page_id: "1"
---
"""
    strategy = FrontMatterIngestion(id_field="page_id")
    chunks = await strategy.ingest(text=text, filename="empty.md", scope=scope)

    assert chunks == []


def test_front_matter_registered_in_registry():
    strategy = get_ingestion_strategy("front_matter")
    assert isinstance(strategy, FrontMatterIngestion)
