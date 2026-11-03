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
