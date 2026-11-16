"""Tests for decision-gated RAG passage filtering."""

from __future__ import annotations

from orchid_ai.agents.rag_pipeline import RagPipeline
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.rag.backends.null import NullVectorReader

PASSAGES = [
    {"text": "billing policy", "score": 0.9},
    {"text": "unrelated weather", "score": 0.8},
    {"text": "refund process", "score": 0.7},
    {"text": "outage status", "score": 0.6},
    {"text": "pizza recipe", "score": 0.5},
]


def _pipeline(provider=None, threshold: float = 0.75) -> RagPipeline:
    return RagPipeline(NullVectorReader(), decision_model=provider, confidence_threshold=threshold)


async def test_relevant_passages_are_kept_in_order() -> None:
    provider = StubDecisionModel(
        answers={
            "chunk:0": {"probability": 0.95},
            "chunk:1": {"probability": 0.1},
            "chunk:2": {"probability": 0.9},
            "chunk:3": {"probability": 0.2},
            "chunk:4": {"probability": 0.1},
        }
    )
    result = await _pipeline(provider)._gate_passages("refund", PASSAGES)
    assert [p["text"] for p in result] == ["billing policy", "refund process"]


async def test_low_confidence_keeps_the_original_ranking() -> None:
    result = await _pipeline(StubDecisionModel(mode="low_confidence"), threshold=0.9)._gate_passages("refund", PASSAGES)
    assert result == PASSAGES


async def test_provider_failure_keeps_the_original_ranking() -> None:
    result = await _pipeline(StubDecisionModel(mode="fail"))._gate_passages("refund", PASSAGES)
    assert result == PASSAGES


async def test_no_provider_keeps_the_original_ranking() -> None:
    result = await _pipeline(None)._gate_passages("refund", PASSAGES)
    assert result == PASSAGES


async def test_too_many_passages_keeps_the_original_ranking() -> None:
    passages = [{"text": f"passage {i}", "score": 1.0} for i in range(33)]
    result = await _pipeline(StubDecisionModel())._gate_passages("query", passages)
    assert result == passages


async def test_empty_selection_keeps_the_original_ranking() -> None:
    provider = StubDecisionModel(answers={f"chunk:{i}": {"probability": 0.1} for i in range(5)})
    result = await _pipeline(provider)._gate_passages("refund", PASSAGES)
    assert result == PASSAGES


class RecordingProvider:
    def __init__(self) -> None:
        self.state = None

    async def decide(self, state, questions, *, timeout=None):
        from orchid_ai.core.decision import OrchidDecisionResult

        self.state = state
        return OrchidDecisionResult(answers={}, provider="recorder", model="recorder")

    async def aclose(self):
        return None


async def test_gate_reads_the_pipeline_content_key() -> None:
    provider = StubDecisionModel(answers={"chunk:0": {"probability": 0.95}, "chunk:1": {"probability": 0.1}})
    passages = [{"content": "billing policy", "score": 0.9}, {"content": "pizza", "score": 0.8}]
    result = await _pipeline(provider)._gate_passages("refund", passages)
    assert [p["content"] for p in result] == ["billing policy"]


async def test_gate_payload_includes_source_and_score() -> None:
    provider = RecordingProvider()
    passages = [{"content": "x", "score": 0.8, "metadata": {"source": "knowledge-base"}}]
    await _pipeline(provider)._gate_passages("q", passages)
    assert provider.state is not None
    payload = provider.state.payload["passages"][0]
    assert payload["source"] == "knowledge-base"
    assert payload["score"] == 0.8


async def test_gate_skips_strategies_not_in_the_allowlist() -> None:
    provider = StubDecisionModel(answers={"chunk:0": {"probability": 0.95}})
    passages = [{"content": "x", "score": 0.9}]
    pipeline = RagPipeline(NullVectorReader(), decision_model=provider, strategies=["simple"])
    assert await pipeline._gate_passages("q", passages, "hyde") == passages


async def test_gate_runs_for_an_allowed_strategy() -> None:
    provider = StubDecisionModel(answers={"chunk:0": {"probability": 0.95}, "chunk:1": {"probability": 0.1}})
    passages = [{"content": "a", "score": 0.9}, {"content": "b", "score": 0.8}]
    pipeline = RagPipeline(NullVectorReader(), decision_model=provider, strategies=["hyde"])
    result = await pipeline._gate_passages("q", passages, "hyde")
    assert result == passages[:1]


class _FakeGraphStore:
    is_null = False

    def __init__(self) -> None:
        from orchid_ai.core.graph_store import OrchidEdge, OrchidEntity

        self.entities = [
            OrchidEntity(id="e1", type="person", name="Alice"),
            OrchidEntity(id="e2", type="person", name="Bob"),
            OrchidEntity(id="e3", type="topic", name="Billing"),
        ]
        self.edges = [
            OrchidEdge(source_id="e1", target_id="e3", relation="owns"),
            OrchidEdge(source_id="e2", target_id="e3", relation="owns"),
        ]

    async def find_entities(self, *, query, scope, k):
        return self.entities[:1]

    async def neighbours(self, ids, *, scope, max_hops, relation_filter):
        return self.entities, self.edges


async def test_graph_strategy_selects_nodes_before_serialisation() -> None:
    from orchid_ai.core.scopes import OrchidRAGScope
    from orchid_ai.rag.strategies.graph_rag import GraphRAGRetrieval

    provider = StubDecisionModel(
        answers={
            "entity:e1": {"probability": 0.95},
            "entity:e2": {"probability": 0.05},
            "entity:e3": {"probability": 0.9},
        }
    )
    strategy = GraphRAGRetrieval(max_hops=1, fuse_with_vectors=False)
    strategy.bind_decision_model(provider, confidence_threshold=0.8)
    results = await strategy.retrieve(
        query="billing",
        namespace="ns",
        scope=OrchidRAGScope(tenant_id="t", user_id="u", chat_id="c", agent_id="a"),
        k=5,
        reader=NullVectorReader(),
        graph_store=_FakeGraphStore(),
    )
    assert results
    assert results[0].document.metadata["entity_count"] == 2
    assert "Alice" in results[0].document.page_content
    assert "Bob" not in results[0].document.page_content


async def test_graph_strategy_fails_open_on_provider_failure() -> None:
    from orchid_ai.core.scopes import OrchidRAGScope
    from orchid_ai.rag.strategies.graph_rag import GraphRAGRetrieval

    strategy = GraphRAGRetrieval(max_hops=1, fuse_with_vectors=False)
    strategy.bind_decision_model(StubDecisionModel(mode="fail"), confidence_threshold=0.8)
    results = await strategy.retrieve(
        query="billing",
        namespace="ns",
        scope=OrchidRAGScope(tenant_id="t", user_id="u", chat_id="c", agent_id="a"),
        k=5,
        reader=NullVectorReader(),
        graph_store=_FakeGraphStore(),
    )
    assert results[0].document.metadata["entity_count"] == 3
