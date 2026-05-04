"""Tests for ``LLMEntityExtractor`` and ``EntityExtractionPostProcessor`` (ADR-026)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.ingestion import OrchidChunk
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.post_processors.entity_extraction import (
    EntityExtractionPostProcessor,
    LLMEntityExtractor,
    OrchidExtractedGraph,
    _ExtractedEdge,
    _ExtractedEntity,
)
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore
from orchid_ai.rag.backends.null import NullGraphStore


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _make_chat_model(extraction: OrchidExtractedGraph) -> MagicMock:
    """Mock a chat model that, when bound via ``with_structured_output``,
    returns the supplied extraction payload from ``ainvoke``."""
    chat_model = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=extraction)
    chat_model.with_structured_output = MagicMock(return_value=structured)
    return chat_model


class TestLLMEntityExtractor:
    @pytest.mark.asyncio
    async def test_extracts_entities_and_edges(self):
        extraction = OrchidExtractedGraph(
            entities=[
                _ExtractedEntity(id="supplier:acme", type="supplier", name="ACME Corp"),
                _ExtractedEntity(id="product:widget", type="product", name="Widget"),
            ],
            edges=[
                _ExtractedEdge(
                    source_id="supplier:acme",
                    target_id="product:widget",
                    relation="supplies",
                ),
            ],
        )
        chat = _make_chat_model(extraction)
        entities, edges = await LLMEntityExtractor().extract(
            "ACME Corp supplies Widget.",
            chat_model=chat,
        )
        assert {e.id for e in entities} == {"supplier:acme", "product:widget"}
        assert len(edges) == 1
        assert edges[0].relation == "supplies"

    @pytest.mark.asyncio
    async def test_drops_dangling_edges(self):
        extraction = OrchidExtractedGraph(
            entities=[_ExtractedEntity(id="a", type="x", name="A")],
            edges=[
                # b is not in entities list — must be dropped.
                _ExtractedEdge(source_id="a", target_id="b", relation="knows"),
            ],
        )
        chat = _make_chat_model(extraction)
        entities, edges = await LLMEntityExtractor().extract("a knows b", chat_model=chat)
        assert {e.id for e in entities} == {"a"}
        assert edges == []  # dangling edge dropped

    @pytest.mark.asyncio
    async def test_empty_text_short_circuits(self):
        chat = _make_chat_model(OrchidExtractedGraph())
        entities, edges = await LLMEntityExtractor().extract("", chat_model=chat)
        assert entities == []
        assert edges == []
        # The LLM was never invoked.
        chat.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chat_model_returns_empty(self):
        entities, edges = await LLMEntityExtractor().extract("text", chat_model=None)
        assert entities == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self):
        chat = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        chat.with_structured_output = MagicMock(return_value=structured)
        entities, edges = await LLMEntityExtractor().extract("text", chat_model=chat)
        assert entities == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_schema_constraints_appear_in_prompt(self):
        """When ``schema={'entity_types': [...]}`` is provided, the
        constraints land in the prompt sent to the LLM."""
        chat = _make_chat_model(OrchidExtractedGraph())
        await LLMEntityExtractor().extract(
            "text",
            chat_model=chat,
            schema={"entity_types": ["supplier", "product"], "relations": ["supplies"]},
        )
        call_args = chat.with_structured_output.return_value.ainvoke.await_args
        sys_message = call_args.args[0][0].content
        assert "supplier, product" in sys_message
        assert "supplies" in sys_message


class TestEntityExtractionPostProcessor:
    @pytest.mark.asyncio
    async def test_writes_to_graph_store_and_tags_chunks(self):
        graph_store = InMemoryGraphStore()
        scope = _scope()
        extraction = OrchidExtractedGraph(
            entities=[
                _ExtractedEntity(id="a", type="x", name="A"),
                _ExtractedEntity(id="b", type="x", name="B"),
            ],
            edges=[_ExtractedEdge(source_id="a", target_id="b", relation="knows")],
        )
        chat = _make_chat_model(extraction)

        chunk = OrchidChunk(text="A knows B.", metadata={"chunk_index": 0})
        out = await EntityExtractionPostProcessor().process(
            [chunk],
            text="A knows B.",
            filename="x.txt",
            chat_model=chat,
            graph_store=graph_store,
            scope=scope,
        )
        assert out[0].metadata["mentioned_entities"] == ["a", "b"]
        # The graph store received both entities + the edge.
        ents, edges = await graph_store.neighbours(["a"], scope=scope, max_hops=1)
        assert {e.id for e in ents} == {"a", "b"}
        assert len(edges) == 1

    @pytest.mark.asyncio
    async def test_no_chat_model_passes_chunks_through(self):
        chunk = OrchidChunk(text="hello", metadata={})
        out = await EntityExtractionPostProcessor().process([chunk], text="hello", filename="x.txt", chat_model=None)
        assert out == [chunk]

    @pytest.mark.asyncio
    async def test_null_graph_store_short_circuits(self):
        """No-op fallback: when graph store is NullGraphStore, no LLM cost."""
        chat = _make_chat_model(OrchidExtractedGraph())
        chunk = OrchidChunk(text="hello", metadata={})
        out = await EntityExtractionPostProcessor().process(
            [chunk],
            text="hello",
            filename="x.txt",
            chat_model=chat,
            graph_store=NullGraphStore(),
            scope=_scope(),
        )
        assert out == [chunk]
        chat.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty(self):
        out = await EntityExtractionPostProcessor().process([], text="hello", filename="x.txt", chat_model=None)
        assert out == []
