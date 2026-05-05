"""Tests for the retrieval strategy registry."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.doc_store import OrchidDocStore
from orchid_ai.core.graph_store import OrchidGraphStore
from orchid_ai.core.repository import OrchidSearchResult, OrchidVectorReader
from orchid_ai.core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.config.schema import OrchidRetrievalConfig
from orchid_ai.config.schema_rag import OrchidHydeConfig
from orchid_ai.rag.strategies import (
    RETRIEVAL_REGISTRY,
    GraphRAGRetrieval,
    HybridRetrieval,
    HyDERetrieval,
    MultiQueryRetrieval,
    SimpleRetrieval,
    clear_retrieval_strategies,
    get_retrieval_strategy,
    register_retrieval_strategy,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_retrieval_strategies()
    yield
    clear_retrieval_strategies()


class _NoopStrategy(OrchidRetrievalStrategy):
    async def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        scope: OrchidRAGScope,
        k: int,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        doc_store: OrchidDocStore | None = None,
        transformers: list[OrchidQueryTransformer] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        return []


class TestBuiltins:
    def test_simple_registered(self):
        assert "simple" in RETRIEVAL_REGISTRY
        assert isinstance(get_retrieval_strategy("simple"), SimpleRetrieval)

    def test_multi_query_registered(self):
        assert "multi_query" in RETRIEVAL_REGISTRY
        assert isinstance(get_retrieval_strategy("multi_query"), MultiQueryRetrieval)

    def test_hyde_registered(self):
        assert "hyde" in RETRIEVAL_REGISTRY
        assert isinstance(get_retrieval_strategy("hyde"), HyDERetrieval)

    def test_hybrid_registered(self):
        assert "hybrid" in RETRIEVAL_REGISTRY
        assert isinstance(get_retrieval_strategy("hybrid"), HybridRetrieval)

    def test_graph_rag_registered(self):
        assert "graph_rag" in RETRIEVAL_REGISTRY
        assert isinstance(get_retrieval_strategy("graph_rag"), GraphRAGRetrieval)


class TestFromConfig:
    def test_returns_default_when_no_config(self):
        strategy = get_retrieval_strategy("hyde")
        assert isinstance(strategy, HyDERetrieval)
        assert strategy._n_hypothetical == 1

    def test_passes_config_through_from_config(self):
        cfg = OrchidRetrievalConfig(hyde=OrchidHydeConfig(n_hypothetical=5))
        strategy = get_retrieval_strategy("hyde", config=cfg)
        assert isinstance(strategy, HyDERetrieval)
        assert strategy._n_hypothetical == 5

    def test_simple_strategy_ignores_config(self):
        cfg = OrchidRetrievalConfig(hyde=OrchidHydeConfig(n_hypothetical=5))
        strategy = get_retrieval_strategy("simple", config=cfg)
        assert isinstance(strategy, SimpleRetrieval)


class TestRegistration:
    def test_register_and_get(self):
        register_retrieval_strategy("noop", _NoopStrategy)
        assert isinstance(get_retrieval_strategy("noop"), _NoopStrategy)

    def test_overwrite_logs_warning(self, caplog):
        with caplog.at_level("WARNING"):
            register_retrieval_strategy("simple", _NoopStrategy)
        assert any("simple" in rec.message for rec in caplog.records)

    def test_unknown_falls_back_to_simple(self):
        assert isinstance(get_retrieval_strategy("does-not-exist"), SimpleRetrieval)


class TestClear:
    def test_clear_restores_builtins(self):
        register_retrieval_strategy("custom", _NoopStrategy)
        assert "custom" in RETRIEVAL_REGISTRY
        clear_retrieval_strategies()
        assert "custom" not in RETRIEVAL_REGISTRY
        assert "simple" in RETRIEVAL_REGISTRY
        assert "multi_query" in RETRIEVAL_REGISTRY
        assert "hyde" in RETRIEVAL_REGISTRY
        assert "hybrid" in RETRIEVAL_REGISTRY
        assert "graph_rag" in RETRIEVAL_REGISTRY
