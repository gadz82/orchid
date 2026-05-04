"""Tests for the new RAG config shape (ADR-022 / ADR-023 / ADR-027)."""

from __future__ import annotations

import pytest

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    OrchidIngestionConfig,
    OrchidRAGConfig,
    OrchidRAGDefaultsConfig,
    OrchidRetrievalConfig,
)


class TestOrchidIngestionConfigDefaults:
    def test_default_strategy_is_none(self):
        """``None`` means 'inherit from defaults'; merger sets it to 'recursive'."""
        cfg = OrchidIngestionConfig()
        assert cfg.strategy is None
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.parent_chunk_size == 0
        assert cfg.parent_chunk_overlap == 200
        assert cfg.post_processors == []


class TestOrchidRetrievalConfigDefaults:
    def test_defaults_inherit(self):
        cfg = OrchidRetrievalConfig()
        assert cfg.strategy is None
        assert cfg.query_transformers is None
        assert cfg.metadata_filters == {}


class TestApplyDefaults:
    def test_unset_strategy_resolves_to_simple(self):
        config = OrchidAgentsConfig(
            agents={"a": OrchidAgentConfig(description="A", prompt="a")},
        )
        agent = config.agents["a"]
        assert agent.rag.retrieval.strategy == "simple"
        assert agent.rag.ingestion.strategy == "recursive"
        assert agent.rag.retrieval.query_transformers == []

    def test_defaults_propagate_strategy(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(strategy="multi_query"),
                ),
            ),
            agents={"a": OrchidAgentConfig(description="A", prompt="a")},
        )
        assert config.agents["a"].rag.retrieval.strategy == "multi_query"

    def test_defaults_propagate_query_transformers(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(query_transformers=["reformulate"]),
                ),
            ),
            agents={"a": OrchidAgentConfig(description="A", prompt="a")},
        )
        assert config.agents["a"].rag.retrieval.query_transformers == ["reformulate"]

    def test_agent_override_strategy_wins(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(strategy="multi_query"),
                ),
            ),
            agents={
                "a": OrchidAgentConfig(
                    description="A",
                    prompt="a",
                    rag=OrchidRAGConfig(retrieval=OrchidRetrievalConfig(strategy="simple")),
                )
            },
        )
        assert config.agents["a"].rag.retrieval.strategy == "simple"

    def test_agent_override_transformers_wins(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(
                    retrieval=OrchidRetrievalConfig(query_transformers=["reformulate"]),
                ),
            ),
            agents={
                "a": OrchidAgentConfig(
                    description="A",
                    prompt="a",
                    rag=OrchidRAGConfig(retrieval=OrchidRetrievalConfig(query_transformers=[])),
                )
            },
        )
        assert config.agents["a"].rag.retrieval.query_transformers == []

    def test_legacy_retriever_type_no_longer_accepted(self):
        """The dropped ``retriever_type`` field is rejected by Pydantic."""
        with pytest.raises(Exception):  # pydantic validation error  # noqa: B017
            OrchidRAGConfig(retriever_type="multi_query")  # type: ignore[call-arg]

    def test_legacy_reformulate_queries_no_longer_accepted(self):
        """The dropped ``reformulate_queries`` field is rejected by Pydantic."""
        with pytest.raises(Exception):  # noqa: B017
            OrchidRAGConfig(reformulate_queries=True)  # type: ignore[call-arg]


class TestYamlRoundTrip:
    def test_yaml_shape(self):
        raw = {
            "defaults": {
                "rag": {
                    "retrieval": {
                        "strategy": "multi_query",
                        "query_transformers": ["reformulate"],
                    }
                }
            },
            "agents": {
                "a": {"description": "A", "prompt": "a"},
                "b": {
                    "description": "B",
                    "prompt": "b",
                    "rag": {"retrieval": {"strategy": "simple"}},
                },
            },
        }
        config = OrchidAgentsConfig(**raw)
        assert config.agents["a"].rag.retrieval.strategy == "multi_query"
        assert config.agents["a"].rag.retrieval.query_transformers == ["reformulate"]
        assert config.agents["b"].rag.retrieval.strategy == "simple"
        # Inherits transformers from defaults — agent only overrode strategy.
        assert config.agents["b"].rag.retrieval.query_transformers == ["reformulate"]
