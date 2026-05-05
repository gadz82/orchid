"""Tests for the ingestion strategy + post-processor registries (ADR-022)."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.ingestion import OrchidChunk, OrchidChunkPostProcessor, OrchidIngestionStrategy
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.documents.post_processors.contextual_headers import (
    ContextualHeaderPostProcessor,
)
from orchid_ai.documents.strategies import (
    INGESTION_REGISTRY,
    POST_PROCESSOR_REGISTRY,
    HeaderedIngestion,
    HierarchicalIngestion,
    RecursiveIngestion,
    SemanticIngestion,
    clear_ingestion_strategies,
    clear_post_processors,
    get_ingestion_strategy,
    get_post_processor,
    register_ingestion_strategy,
    register_post_processor,
)


@pytest.fixture(autouse=True)
def _reset_registries():
    clear_ingestion_strategies()
    clear_post_processors()
    yield
    clear_ingestion_strategies()
    clear_post_processors()


class _NoopStrategy(OrchidIngestionStrategy):
    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: Any | None = None,
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        return [OrchidChunk(text=text, metadata={"source": filename})]


class _UpperHeader(OrchidChunkPostProcessor):
    async def process(self, chunks, *, text, filename, chat_model=None):
        return [OrchidChunk(text=c.text.upper(), metadata=c.metadata) for c in chunks]


class TestIngestionRegistry:
    def test_recursive_built_in(self):
        assert "recursive" in INGESTION_REGISTRY
        assert isinstance(get_ingestion_strategy("recursive"), RecursiveIngestion)

    def test_semantic_built_in(self):
        assert "semantic" in INGESTION_REGISTRY
        assert isinstance(get_ingestion_strategy("semantic"), SemanticIngestion)

    def test_hierarchical_built_in(self):
        assert "hierarchical" in INGESTION_REGISTRY
        assert isinstance(get_ingestion_strategy("hierarchical"), HierarchicalIngestion)

    def test_headered_built_in(self):
        assert "headered" in INGESTION_REGISTRY
        assert isinstance(get_ingestion_strategy("headered"), HeaderedIngestion)

    def test_register_and_get(self):
        register_ingestion_strategy("noop", _NoopStrategy)
        assert isinstance(get_ingestion_strategy("noop"), _NoopStrategy)

    def test_unknown_falls_back_to_recursive(self):
        assert isinstance(get_ingestion_strategy("missing"), RecursiveIngestion)

    def test_clear_restores_builtins(self):
        register_ingestion_strategy("custom", _NoopStrategy)
        clear_ingestion_strategies()
        assert "custom" not in INGESTION_REGISTRY
        assert "recursive" in INGESTION_REGISTRY
        assert "semantic" in INGESTION_REGISTRY
        assert "hierarchical" in INGESTION_REGISTRY
        assert "headered" in INGESTION_REGISTRY


class TestPostProcessorRegistry:
    def test_contextual_headers_built_in(self):
        assert "contextual_headers" in POST_PROCESSOR_REGISTRY
        assert isinstance(get_post_processor("contextual_headers"), ContextualHeaderPostProcessor)

    def test_entity_extraction_built_in(self):
        from orchid_ai.documents.post_processors.entity_extraction import (
            EntityExtractionPostProcessor,
        )

        assert "entity_extraction" in POST_PROCESSOR_REGISTRY
        assert isinstance(get_post_processor("entity_extraction"), EntityExtractionPostProcessor)

    def test_register_and_get(self):
        register_post_processor("upper", _UpperHeader)
        assert isinstance(get_post_processor("upper"), _UpperHeader)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown chunk post-processor"):
            get_post_processor("missing")

    def test_clear_restores_builtins(self):
        register_post_processor("upper", _UpperHeader)
        clear_post_processors()
        assert "upper" not in POST_PROCESSOR_REGISTRY
        assert "contextual_headers" in POST_PROCESSOR_REGISTRY
        assert "entity_extraction" in POST_PROCESSOR_REGISTRY
