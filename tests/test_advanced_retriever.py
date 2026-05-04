"""Tests for the new RAG strategy / ingestion / transformer surface
(replaces the prior ``test_advanced_retriever.py`` which targeted the
removed ``multi_query_retrieve`` free function and ``retriever_type``
field)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    OrchidRAGDefaultsConfig,
    OrchidRetrievalConfig,
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


class TestRetrievalConfig:
    """OrchidRAGConfig.retrieval block."""

    def test_default_strategy_resolves_to_simple(self):
        """An agent with no overrides resolves retrieval.strategy to 'simple'."""
        config = OrchidAgentsConfig(
            agents={"test": OrchidAgentConfig(description="t", prompt="p")},
        )
        assert config.agents["test"].rag.retrieval.strategy == "simple"

    def test_defaults_propagate_strategy(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(strategy="multi_query"),
                ),
            ),
            agents={"test": OrchidAgentConfig(description="t", prompt="p")},
        )
        assert config.agents["test"].rag.retrieval.strategy == "multi_query"

    def test_agent_overrides_defaults(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(strategy="multi_query"),
                ),
            ),
            agents={
                "a": OrchidAgentConfig(description="A", prompt="a"),
                "b": OrchidAgentConfig(
                    description="B",
                    prompt="b",
                    rag={"retrieval": {"strategy": "simple"}},  # type: ignore[arg-type]
                ),
            },
        )
        assert config.agents["a"].rag.retrieval.strategy == "multi_query"
        assert config.agents["b"].rag.retrieval.strategy == "simple"

    def test_query_transformers_inherit_from_defaults(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(query_transformers=["reformulate"]),
                ),
            ),
            agents={"test": OrchidAgentConfig(description="t", prompt="p")},
        )
        assert config.agents["test"].rag.retrieval.query_transformers == ["reformulate"]


# ── OrchidRetriever (LangChain BaseRetriever wrapper) ───────


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


# ── Recursive ingestion tests (replace parent_child_chunk_text tests) ─


class TestRecursiveIngestionParentChild:
    """RecursiveIngestion.ingest with parent_chunk_size > 0 produces
    child chunks with parent content stored in metadata."""

    @pytest.mark.asyncio
    async def test_basic_parent_child(self):
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.strategies import RecursiveIngestion

        text = "A" * 3000
        cfg = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            parent_chunk_size=800,
            parent_chunk_overlap=50,
        )

        chunks = await RecursiveIngestion(cfg).ingest(text=text, filename="big.txt", scope=_make_scope())
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.text) <= 220  # chunk_size + tolerance
            assert "parent_content" in c.metadata
            assert len(c.metadata["parent_content"]) <= 820  # parent_chunk_size + tolerance
            assert c.metadata.get("parent_index", -1) >= 0

    @pytest.mark.asyncio
    async def test_empty_text(self):
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.strategies import RecursiveIngestion

        chunks = await RecursiveIngestion(ChunkConfig(parent_chunk_size=500)).ingest(
            text="", filename="empty.txt", scope=_make_scope()
        )
        assert chunks == []

    @pytest.mark.asyncio
    async def test_small_text_single_chunk(self):
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.strategies import RecursiveIngestion

        cfg = ChunkConfig(chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        chunks = await RecursiveIngestion(cfg).ingest(text="Short text", filename="short.txt", scope=_make_scope())
        assert len(chunks) == 1


class TestParentDocumentRetrieval:
    """fetch_rag_context() prefers parent_content from metadata."""

    @pytest.mark.asyncio
    async def test_parent_content_preferred(self):
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
        assert "parent_content" not in result[0]["metadata"]

    @pytest.mark.asyncio
    async def test_no_parent_uses_page_content(self):
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
    """ingest_document() with RecursiveIngestion(parent_chunk_size>0) stores
    parent_content in metadata."""

    @pytest.mark.asyncio
    async def test_parent_child_ingestion(self):
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.pipeline import ingest_document
        from orchid_ai.documents.strategies import RecursiveIngestion

        mock_writer = MagicMock()
        mock_writer.upsert = AsyncMock()

        text = "Word " * 500
        ingestion = RecursiveIngestion(ChunkConfig(chunk_size=200, parent_chunk_size=600))
        count = await ingest_document(
            file_bytes=text.encode(),
            filename="test.txt",
            scope=_make_scope(),
            writer=mock_writer,
            ingestion=ingestion,
            pre_extracted_text=text,
        )

        assert count > 0
        mock_writer.upsert.assert_called_once()
        docs = mock_writer.upsert.call_args[0][0]

        for doc in docs:
            assert "parent_content" in doc.metadata
            assert doc.metadata["parent_content"]
            assert "parent_index" in doc.metadata
            assert len(doc.page_content) <= 220

    @pytest.mark.asyncio
    async def test_standard_ingestion_no_parent(self):
        """parent_chunk_size=0 → flat chunks, no parent metadata."""
        from orchid_ai.documents.chunker import ChunkConfig
        from orchid_ai.documents.pipeline import ingest_document
        from orchid_ai.documents.strategies import RecursiveIngestion

        mock_writer = MagicMock()
        mock_writer.upsert = AsyncMock()

        text = "Word " * 500
        ingestion = RecursiveIngestion(ChunkConfig(chunk_size=200, parent_chunk_size=0))
        count = await ingest_document(
            file_bytes=text.encode(),
            filename="test.txt",
            scope=_make_scope(),
            writer=mock_writer,
            ingestion=ingestion,
            pre_extracted_text=text,
        )

        assert count > 0
        docs = mock_writer.upsert.call_args[0][0]

        for doc in docs:
            assert "parent_content" not in doc.metadata
