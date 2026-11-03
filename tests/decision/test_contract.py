"""Cross-adapter contract suite.

Both wire dialects answer the same logical questions. These tests build
equivalent payloads in each dialect and assert the normalised results agree on
value and confidence.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.decision import OrchidDecisionQuestion, OrchidDecisionState, aggregate_shortlist
from orchid_ai.decision.openai_compatible import OpenAICompatibleDecisionModel
from orchid_ai.decision.state_questions import StateQuestionsDecisionModel

QUESTIONS = [
    OrchidDecisionQuestion(name="relevant", kind="boolean"),
    OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
    OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2", "3")),
]


class FakeClient:
    host = "api.example.com"

    def __init__(self, response: Any) -> None:
        self.response = response
        self.payload: dict[str, Any] | None = None

    async def request_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        self.payload = payload
        return self.response

    async def aclose(self) -> None:
        return None


async def _run_state_questions(response: dict[str, Any], questions=QUESTIONS):
    client = FakeClient(response)
    adapter = StateQuestionsDecisionModel(client=client, model="decision-model-v1", provider="fast")
    return await adapter.decide(OrchidDecisionState(text="x"), questions), client


async def _run_openai(response: dict[str, Any], questions=QUESTIONS):
    client = FakeClient(response)
    adapter = OpenAICompatibleDecisionModel(client=client, model="decision-model-v1", provider="fast")
    return await adapter.decide(OrchidDecisionState(text="x"), questions), client


def _state_questions_payload(
    relevant=0.92, route="billing", route_confidence=0.88, urgency=2.4, urgency_confidence=0.49
):
    return {
        "answers": {
            "relevant": {"type": "noul", "noul": relevant},
            "route": {"type": "choice", "choice": route, "confidence": route_confidence},
            "urgency": {"type": "score", "score": urgency, "confidence": urgency_confidence},
        }
    }


def _openai_payload(relevant=0.92, route="billing", route_confidence=0.88, urgency=2.4, urgency_confidence=0.49):
    return {
        "answers": [
            {"type": "predicate", "name": "relevant", "probability": relevant},
            {"type": "choice", "name": "route", "choice": route, "confidence": route_confidence},
            {"type": "score", "name": "urgency", "score": urgency, "confidence": urgency_confidence},
        ]
    }


async def test_adapters_agree_on_boolean_true() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    for result in (state_questions, openai):
        assert result.answers["relevant"].value is True
        assert result.answers["relevant"].confidence == pytest.approx(0.92)


async def test_adapters_agree_on_confident_false() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload(relevant=0.3))
    openai, _ = await _run_openai(_openai_payload(relevant=0.3))
    for result in (state_questions, openai):
        assert result.answers["relevant"].value is False
        assert result.answers["relevant"].confidence == pytest.approx(0.7)


async def test_adapters_agree_on_choice() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    for result in (state_questions, openai):
        assert result.answers["route"].value == "billing"
        assert result.answers["route"].confidence == pytest.approx(0.88)


async def test_adapters_agree_on_score() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    for result in (state_questions, openai):
        assert result.answers["urgency"].value == pytest.approx(2.4)
        assert result.answers["urgency"].confidence == pytest.approx(0.49)


async def test_adapters_agree_on_refusal() -> None:
    state_questions, _ = await _run_state_questions({"answers": {"relevant": {"type": "refusal"}}}, QUESTIONS[:1])
    openai, _ = await _run_openai({"answers": [{"type": "refusal", "name": "relevant"}]}, QUESTIONS[:1])
    for result in (state_questions, openai):
        answer = result.answers["relevant"]
        assert answer.value is None
        assert answer.refusal
        assert answer.confidence == 0.0


async def test_confident_false_is_not_shortlisted_by_either_adapter() -> None:
    state_questions, _ = await _run_state_questions(
        {"answers": {"relevant": {"type": "noul", "noul": 0.2}}}, QUESTIONS[:1]
    )
    openai, _ = await _run_openai(
        {"answers": [{"type": "predicate", "name": "relevant", "probability": 0.2}]}, QUESTIONS[:1]
    )
    for result in (state_questions, openai):
        assert aggregate_shortlist(result, ["relevant"], threshold=0.5) is None


async def test_both_adapters_ask_every_question() -> None:
    _, state_client = await _run_state_questions(_state_questions_payload())
    _, openai_client = await _run_openai(_openai_payload())
    assert state_client.payload is not None and openai_client.payload is not None
    assert set(state_client.payload["questions"]) == {"relevant", "route", "urgency"}
    assert [question["name"] for question in openai_client.payload["questions"]] == [
        "relevant",
        "route",
        "urgency",
    ]
