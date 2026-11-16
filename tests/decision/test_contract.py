"""Cross-adapter contract suite.

Both wire dialects answer the same logical questions. These tests build
equivalent payloads in each dialect and assert the normalised results are
identical, including probabilities and refusal text.
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
            "route": {
                "type": "choice",
                "choice": route,
                "confidence": route_confidence,
                "probabilities": {"billing": route_confidence, "technical": 1.0 - route_confidence},
            },
            "urgency": {
                "type": "score",
                "score": urgency,
                "confidence": urgency_confidence,
                "probabilities": {"0": 0.01, "1": 0.04, "2": 0.49, "3": 0.46},
            },
        }
    }


def _openai_payload(relevant=0.92, route="billing", route_confidence=0.88, urgency=2.4, urgency_confidence=0.49):
    return {
        "answers": [
            {"type": "predicate", "name": "relevant", "probability": relevant},
            {
                "type": "choice",
                "name": "route",
                "choice": route,
                "confidence": route_confidence,
                "probabilities": [
                    {"value": "billing", "probability": route_confidence},
                    {"value": "technical", "probability": 1.0 - route_confidence},
                ],
            },
            {
                "type": "score",
                "name": "urgency",
                "score": urgency,
                "confidence": urgency_confidence,
                "probabilities": [
                    {"label": "0", "value": 0, "probability": 0.01},
                    {"label": "1", "value": 1, "probability": 0.04},
                    {"label": "2", "value": 2, "probability": 0.49},
                    {"label": "3", "value": 3, "probability": 0.46},
                ],
            },
        ]
    }


async def test_adapters_agree_on_boolean_true() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    assert state_questions.answers == openai.answers
    assert state_questions.answers["relevant"].value is True
    assert state_questions.answers["relevant"].confidence == pytest.approx(0.92)


async def test_adapters_agree_on_confident_false() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload(relevant=0.3))
    openai, _ = await _run_openai(_openai_payload(relevant=0.3))
    assert state_questions.answers == openai.answers
    assert state_questions.answers["relevant"].value is False
    assert state_questions.answers["relevant"].confidence == pytest.approx(0.7)


async def test_adapters_agree_on_choice_and_probabilities() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    assert state_questions.answers["route"] == openai.answers["route"]
    assert state_questions.answers["route"].probabilities == {"billing": 0.88, "technical": 0.12}


async def test_adapters_agree_on_score_and_probabilities() -> None:
    state_questions, _ = await _run_state_questions(_state_questions_payload())
    openai, _ = await _run_openai(_openai_payload())
    assert state_questions.answers["urgency"] == openai.answers["urgency"]
    assert state_questions.answers["urgency"].probabilities["2"] == 0.49


async def test_adapters_agree_on_refusal_reason() -> None:
    state_questions, _ = await _run_state_questions(
        {"answers": {"relevant": {"type": "refusal", "reason": "insufficient context"}}}, QUESTIONS[:1]
    )
    openai, _ = await _run_openai(
        {"answers": [{"type": "refusal", "name": "relevant", "refusal": "insufficient context"}]}, QUESTIONS[:1]
    )
    assert state_questions.answers == openai.answers
    assert state_questions.answers["relevant"].refusal == "insufficient context"


async def test_probability_only_confidence_paths_agree() -> None:
    state_questions, _ = await _run_state_questions(
        {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.77, "technical": 0.23},
                }
            }
        },
        [QUESTIONS[1]],
    )
    openai, _ = await _run_openai(
        {
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "billing",
                    "probabilities": [
                        {"value": "billing", "probability": 0.77},
                        {"value": "technical", "probability": 0.23},
                    ],
                }
            ]
        },
        [QUESTIONS[1]],
    )
    assert state_questions.answers == openai.answers
    assert state_questions.answers["route"].confidence == 0.77


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
