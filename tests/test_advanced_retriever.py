"""Tests for advanced retriever (#8) and parent document retriever (#12)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.documents import Document

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    OrchidRAGConfig,
    OrchidRAGDefaultsConfig,
)
from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.rag.scopes import OrchidRAGScope


# ── Fixtures ────────────────────────────────────────────────


def _make_scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="test")


def _make_search_results(n: int = 3) -> list[OrchidSearchResult]:
    return [
        OrchidSearchResult(
            document=Document(
                id=f"doc-{i}",
                page_content=f"Content of document {i}",
                metadata={"tenant_id": "t1", "chunk_index": i},
            ),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


# ── Schema tests ────────────────────────────────────────────


class TestRetrieverTypeConfig:
    """OrchidRAGConfig.retriever_type field."""

    def test_default_is_none(self):
        cfg = OrchidRAGConfig()
        assert cfg.retriever_type is None  # None = inherit from defaults

    def test_multi_query(self):
        cfg = OrchidRAGConfig(retriever_type="multi_query")
        assert cfg.retriever_type == "multi_query"

    def test_defaults_propagate(self):
        cfg = OrchidRAGDefaultsConfig(retriever_type="multi_query")
        assert cfg.retriever_type == "multi_query"

    def test_agent_inherits_retriever_type(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(retriever_type="multi_query"),
            ),
            agents={
                "test": OrchidAgentConfig(description="test", prompt="test"),
            },
        )
        assert config.agents["test"].rag.retriever_type == "multi_query"

    def test_agent_keeps_simple_when_default_simple(self):
        config = OrchidAgentsConfig(
            agents={
                "test": OrchidAgentConfig(description="test", prompt="test"),
            },
        )
        assert config.agents["test"].rag.retriever_type == "simple"

    def test_yaml_round_trip(self):
        raw = {
            "defaults": {
                "rag": {"retriever_type": "multi_query"},
            },
            "agents": {
                "a": {"description": "A", "prompt": "a"},
                "b": {
                    "description": "B",
                    "prompt": "b",
                    "rag": {"retriever_type": "simple"},  # explicit override
                },
            },
        }
        config = OrchidAgentsConfig(**raw)
        assert config.agents["a"].rag.retriever_type == "multi_query"
        assert config.agents["b"].rag.retriever_type == "simple"


# ── OrchidRetriever tests ───────────────────────────────────


class TestOrchidRetriever:
    """OrchidRetriever wraps OrchidVectorReader as BaseRetriever."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_documents(self):
        from orchid_ai.rag.retriever import OrchidRetriever

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(return_value=_make_search_results(3))

        retriever = OrchidRetriever(
            reader=mock_reader,
            namespace="learning",
            scope=_make_scope(),
            k=5,
        )

        docs = await retriever.ainvoke("test query")
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].page_content == "Content of document 0"

    @pytest.mark.asyncio
    async def test_ainvoke_passes_params(self):
        from orchid_ai.rag.retriever import OrchidRetriever

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(return_value=[])
        scope = _make_scope()

        retriever = OrchidRetriever(
            reader=mock_reader,
            namespace="uploads",
            scope=scope,
            k=10,
        )

        await retriever.ainvoke("hello")
        mock_reader.retrieve.assert_called_once_with(
            query="hello",
            namespace="uploads",
            k=10,
            scope=scope,
        )


# ── Multi-query retrieve tests ──────────────────────────────


class TestMultiQueryRetrieve:
    """multi_query_retrieve() generates variations and deduplicates."""

    @pytest.mark.asyncio
    async def test_merges_results_from_variations(self):
        from orchid_ai.rag.retriever import multi_query_retrieve

        mock_reader = MagicMock()
        scope = _make_scope()

        # First query returns docs 0,1,2; second returns 2,3,4
        call_count = 0

        async def side_effect(query, namespace, k, scope):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_search_results(3)
            return [
                OrchidSearchResult(
                    document=Document(id=f"doc-{i}", page_content=f"Content {i}", metadata={}),
                    score=0.8 - (i - 2) * 0.1,
                )
                for i in range(2, 5)
            ]

        mock_reader.retrieve = AsyncMock(side_effect=side_effect)

        mock_chat = MagicMock()
        mock_chat.ainvoke = AsyncMock(return_value=MagicMock(content="variation 1\nvariation 2"))

        results = await multi_query_retrieve(
            "original query",
            mock_reader,
            "ns",
            scope,
            mock_chat,
            k=5,
            num_queries=1,  # 1 variation + original = 2 queries
        )

        # Should get deduplicated results (doc-0 through doc-4)
        doc_ids = {r.document.id for r in results}
        assert len(results) <= 5
        assert len(doc_ids) == len(results)  # no duplicates

    @pytest.mark.asyncio
    async def test_fallback_on_variation_failure(self):
        from orchid_ai.rag.retriever import multi_query_retrieve

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(return_value=_make_search_results(2))

        # Chat model fails to generate variations
        mock_chat = MagicMock()
        mock_chat.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        results = await multi_query_retrieve(
            "query",
            mock_reader,
            "ns",
            _make_scope(),
            mock_chat,
            k=5,
        )

        # Should still return results from the original query
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_deduplicates_by_id_keeps_highest_score(self):
        from orchid_ai.rag.retriever import multi_query_retrieve

        mock_reader = MagicMock()

        # Both queries return same doc but with different scores
        results_a = [OrchidSearchResult(document=Document(id="d1", page_content="x", metadata={}), score=0.5)]
        results_b = [OrchidSearchResult(document=Document(id="d1", page_content="x", metadata={}), score=0.9)]

        mock_reader.retrieve = AsyncMock(side_effect=[results_a, results_b])

        mock_chat = MagicMock()
        mock_chat.ainvoke = AsyncMock(return_value=MagicMock(content="variation"))

        results = await multi_query_retrieve(
            "q",
            mock_reader,
            "ns",
            _make_scope(),
            mock_chat,
            k=5,
            num_queries=1,
        )

        assert len(results) == 1
        assert results[0].score == 0.9  # kept the higher score


# ── Parent Document Retriever tests ─────────────────────────


class TestParentChildChunking:
    """parent_child_chunk_text() produces child chunks with parent refs."""

    def test_basic_parent_child(self):
        from orchid_ai.documents.chunker import ChunkConfig, parent_child_chunk_text

        text = "A" * 3000  # enough for multiple parent + child chunks
        cfg = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            parent_chunk_size=800,
            parent_chunk_overlap=50,
        )

        result = parent_child_chunk_text(text, cfg)
        assert len(result) > 0

        # Each child chunk is smaller than its parent
        for pc in result:
            assert len(pc.child_text) <= 220  # chunk_size + some tolerance
            assert len(pc.parent_text) <= 820  # parent_chunk_size + some tolerance
            assert pc.parent_index >= 0
            assert pc.child_index >= 0

    def test_empty_text(self):
        from orchid_ai.documents.chunker import ChunkConfig, parent_child_chunk_text

        cfg = ChunkConfig(parent_chunk_size=500)
        assert parent_child_chunk_text("", cfg) == []
        assert parent_child_chunk_text("   ", cfg) == []

    def test_small_text_single_chunk(self):
        from orchid_ai.documents.chunker import ChunkConfig, parent_child_chunk_text

        cfg = ChunkConfig(chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        result = parent_child_chunk_text("Short text", cfg)
        assert len(result) == 1
        assert result[0].child_text.strip() == "Short text"
        assert result[0].parent_text.strip() == "Short text"


class TestParentDocumentRetrieval:
    """fetch_rag_context() prefers parent_content from metadata."""

    @pytest.mark.asyncio
    async def test_parent_content_preferred(self):
        """When parent_content is in metadata, it's used as content."""
        from orchid_ai.core.agent import OrchidAgent

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(
            return_value=[
                OrchidSearchResult(
                    document=Document(
                        id="d1",
                        page_content="child chunk text",
                        metadata={
                            "parent_content": "full parent paragraph with more context",
                            "tenant_id": "t1",
                            "parent_index": 0,
                        },
                    ),
                    score=0.95,
                ),
            ]
        )

        class _TestAgent(OrchidAgent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test agent"

            async def run(self, state):
                pass

        agent = _TestAgent(model_id="test", reader=mock_reader)
        result = await agent.fetch_rag_context("query", _make_scope())

        assert len(result) == 1
        assert result[0]["content"] == "full parent paragraph with more context"
        # parent_content should NOT be in metadata dict
        assert "parent_content" not in result[0]["metadata"]

    @pytest.mark.asyncio
    async def test_no_parent_uses_page_content(self):
        """Without parent_content, regular page_content is used."""
        from orchid_ai.core.agent import OrchidAgent

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(
            return_value=[
                OrchidSearchResult(
                    document=Document(
                        id="d1",
                        page_content="regular chunk text",
                        metadata={"tenant_id": "t1"},
                    ),
                    score=0.9,
                ),
            ]
        )

        class _TestAgent(OrchidAgent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test agent"

            async def run(self, state):
                pass

        agent = _TestAgent(model_id="test", reader=mock_reader)
        result = await agent.fetch_rag_context("query", _make_scope())

        assert len(result) == 1
        assert result[0]["content"] == "regular chunk text"


class TestParentDocumentIngestion:
    """ingest_document() stores parent_content in metadata."""

    @pytest.mark.asyncio
    async def test_parent_child_ingestion(self):
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.pipeline import ingest_document

        mock_writer = MagicMock()
        mock_writer.upsert = AsyncMock()

        text = "Word " * 500  # ~2500 chars
        count = await ingest_document(
            file_bytes=text.encode(),
            filename="test.txt",
            scope=_make_scope(),
            writer=mock_writer,
            chunk_config=ChunkConfig(
                chunk_size=200,
                parent_chunk_size=600,
            ),
            pre_extracted_text=text,
        )

        assert count > 0
        mock_writer.upsert.assert_called_once()
        docs = mock_writer.upsert.call_args[0][0]

        # Every document should have parent_content in metadata
        for doc in docs:
            assert "parent_content" in doc.metadata
            assert doc.metadata["parent_content"]  # non-empty
            assert "parent_index" in doc.metadata
            assert len(doc.page_content) <= 220  # child chunk size

    @pytest.mark.asyncio
    async def test_standard_ingestion_no_parent(self):
        """With parent_chunk_size=0, standard chunking (no parent metadata)."""
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.pipeline import ingest_document

        mock_writer = MagicMock()
        mock_writer.upsert = AsyncMock()

        text = "Word " * 500
        count = await ingest_document(
            file_bytes=text.encode(),
            filename="test.txt",
            scope=_make_scope(),
            writer=mock_writer,
            chunk_config=ChunkConfig(chunk_size=200, parent_chunk_size=0),
            pre_extracted_text=text,
        )

        assert count > 0
        docs = mock_writer.upsert.call_args[0][0]

        for doc in docs:
            assert "parent_content" not in doc.metadata
