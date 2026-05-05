"""Tests for ``SemanticIngestion`` (ADR-022)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.strategies.semantic import SemanticIngestion


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")


def _topic_aligned_embeddings(sentences: list[str]) -> list[list[float]]:
    """Build deterministic mock embeddings — sentences containing the
    word ``alpha`` map to vector ``[1, 0, 0]`` (topic A) and sentences
    containing ``beta`` map to ``[0, 1, 0]`` (topic B).  All other
    sentences map to ``[0.5, 0.5, 0]`` (ambiguous).
    """
    out: list[list[float]] = []
    for s in sentences:
        lower = s.lower()
        if "alpha" in lower:
            out.append([1.0, 0.0, 0.0])
        elif "beta" in lower:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.5, 0.5, 0.0])
    return out


def _make_embedder(sentences: list[str]):
    embedder = MagicMock()
    embedder.aembed_documents = AsyncMock(side_effect=lambda texts: _topic_aligned_embeddings(texts))
    return embedder


class TestSemanticIngestion:
    @pytest.mark.asyncio
    async def test_falls_back_when_no_embedder(self):
        """No embedder → recursive fallback (still produces chunks)."""
        text = "First sentence. Second sentence. Third sentence." * 30
        chunks = await SemanticIngestion().ingest(text=text, filename="t.txt", scope=_scope(), embeddings=None)
        assert chunks
        for c in chunks:
            # Recursive fallback doesn't tag with the semantic strategy marker
            assert c.metadata.get("ingestion_strategy") != "semantic"

    @pytest.mark.asyncio
    async def test_falls_back_when_text_too_short(self):
        """Below ``min_chunk_chars`` → recursive fallback regardless of embedder."""
        text = "Tiny."
        embedder = MagicMock()
        embedder.aembed_documents = AsyncMock()
        chunks = await SemanticIngestion(min_chunk_chars=500).ingest(
            text=text, filename="t.txt", scope=_scope(), embeddings=embedder
        )
        # Recursive fallback returns 1 chunk; embedder was never invoked.
        assert chunks
        embedder.aembed_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chunk_crosses_topic_switch(self):
        """Boundary detection isolates topic A from topic B."""
        # 6 sentences: 3 about topic alpha, 3 about topic beta.
        text = (
            "The alpha protocol governs alpha networks. "
            "Alpha messages encode alpha state. "
            "Alpha runtime managers coordinate alpha workers. "
            "The beta protocol secures beta packets. "
            "Beta tokens authenticate beta peers. "
            "Beta orchestrators schedule beta tasks."
        )
        embedder = _make_embedder(_scope_split_sentences(text))

        chunks = await SemanticIngestion(min_chunk_chars=10, breakpoint_percentile=50).ingest(
            text=text, filename="t.txt", scope=_scope(), embeddings=embedder
        )

        assert chunks
        for c in chunks:
            text_lower = c.text.lower()
            # Each chunk must be on exactly one side of the topic switch.
            assert not ("alpha" in text_lower and "beta" in text_lower), f"Chunk crossed topic boundary: {c.text!r}"

    @pytest.mark.asyncio
    async def test_marks_metadata_with_strategy_name(self):
        text = (
            "Alpha words start the document. "
            "More alpha discussion follows. "
            "Beta material continues here. "
            "Final beta paragraph wraps it up."
        )
        embedder = _make_embedder(_scope_split_sentences(text))
        chunks = await SemanticIngestion(min_chunk_chars=10, breakpoint_percentile=50).ingest(
            text=text, filename="t.txt", scope=_scope(), embeddings=embedder
        )
        assert chunks
        for c in chunks:
            assert c.metadata["ingestion_strategy"] == "semantic"
            assert c.metadata["source_file"] == "t.txt"

    @pytest.mark.asyncio
    async def test_embedder_failure_falls_back_to_recursive(self):
        """A raising embedder doesn't crash ingestion."""
        text = "This is a long enough document. " * 50
        embedder = MagicMock()
        embedder.aembed_documents = AsyncMock(side_effect=RuntimeError("embed boom"))

        chunks = await SemanticIngestion(min_chunk_chars=10).ingest(
            text=text, filename="t.txt", scope=_scope(), embeddings=embedder
        )
        assert chunks  # recursive fallback still chunks the text
        for c in chunks:
            assert c.metadata.get("ingestion_strategy") != "semantic"

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        embedder = _make_embedder([])
        assert await SemanticIngestion().ingest(text="", filename="t.txt", scope=_scope(), embeddings=embedder) == []
        assert (
            await SemanticIngestion().ingest(text="   \n", filename="t.txt", scope=_scope(), embeddings=embedder) == []
        )

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="breakpoint_percentile"):
            SemanticIngestion(breakpoint_percentile=0.0)
        with pytest.raises(ValueError, match="breakpoint_percentile"):
            SemanticIngestion(breakpoint_percentile=100.0)


def _scope_split_sentences(text: str) -> list[str]:
    """Re-export the strategy's internal sentence splitter for the test
    helper to align the mock embeddings list length."""
    from orchid_ai.documents.strategies.semantic import _split_sentences

    return _split_sentences(text)
