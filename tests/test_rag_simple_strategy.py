"""Tests for ``SimpleRetrieval`` (ADR-023)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.strategies.simple import SimpleRetrieval


@pytest.mark.asyncio
async def test_simple_retrieval_passes_through_to_reader():
    expected = [
        OrchidSearchResult(
            document=Document(id="d1", page_content="hello", metadata={"x": 1}),
            score=0.9,
        )
    ]
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=expected)

    scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")
    results = await SimpleRetrieval().retrieve(query="q", namespace="kb", scope=scope, k=3, reader=reader)

    assert results == expected
    reader.retrieve.assert_called_once_with(query="q", namespace="kb", k=3, scope=scope, metadata_filters=None)


@pytest.mark.asyncio
async def test_simple_retrieval_ignores_optional_kwargs():
    """SimpleRetrieval is the default — optional graph / doc / transformer
    args must not break the call."""
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])

    scope = OrchidRAGScope(tenant_id="t")
    results = await SimpleRetrieval().retrieve(
        query="q",
        namespace="kb",
        scope=scope,
        k=5,
        reader=reader,
        chat_model=MagicMock(),
        graph_store=MagicMock(),
        doc_store=MagicMock(),
        transformers=[],
        metadata_filters={"status": "published"},
    )
    assert results == []
