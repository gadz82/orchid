"""Tests for the state-and-questions decision adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from orchid_ai.core.decision import (
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from orchid_ai.decision.state_questions import StateQuestionsDecisionModel

FIXTURES = Path(__file__).parent / "fixtures" / "state_questions.json"


class FakeClient:
    host = "api.example.com"

    def __init__(self, response: Any) -> None:
        self.response = response
        self.path: str | None = None
        self.payload: dict[str, Any] | None = None
        self.timeout: float | None = None
        self.closed = False

    async def request_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        self.path = path
        self.payload = payload
        self.timeout = timeout
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def _adapter(response: Any) -> tuple[StateQuestionsDecisionModel, FakeClient]:
    client = FakeClient(response)
    return StateQuestionsDecisionModel(client=client, model="decision-model-v1", provider="fast"), client


def _fixture(name: str) -> dict[str, Any]:
    return json.loads(FIXTURES.read_text())[name]


async def test_request_shape_for_all_three_kinds() -> None:
    adapter, client = _adapter({"answers": {}})
    questions = [
        OrchidDecisionQuestion(name="relevant", kind="boolean", instructions="Is it relevant?", required=False),
        OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"), required=False),
        OrchidDecisionQuestion(name="urgency", kind="score", rubric=("low", "medium", "high"), required=False),
    ]
    await adapter.decide(OrchidDecisionState(text="my invoice is wrong"), questions)

    assert client.path == "/v1/decide"
    assert client.payload is not None
    assert client.payload["model"] == "decision-model-v1"
    assert client.payload["state"] == "my invoice is wrong"
    assert client.payload["questions"] == {
        "relevant": {"type": "noul", "instructions": "Is it relevant?"},
        "route": {"type": "choice", "criteria": {"billing": "billing", "technical": "technical"}},
        "urgency": {"type": "score", "criteria": ["low", "medium", "high"]},
    }


async def test_state_with_payload_is_sent_as_text_and_context() -> None:
    adapter, client = _adapter({"answers": {}})
    state = OrchidDecisionState(text="pick one", payload={"candidates": ["a", "b"]})
    await adapter.decide(state, [OrchidDecisionQuestion(name="q", kind="boolean", required=False)])
    assert client.payload is not None
    assert client.payload["state"] == {"text": "pick one", "context": {"candidates": ["a", "b"]}}


async def test_boolean_normalisation() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 0.3}}})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    answer = result.answers["q"]
    assert answer.value is False
    assert answer.confidence == pytest.approx(0.7)
    assert answer.probabilities == {"true": 0.3, "false": pytest.approx(0.7)}


async def test_choice_normalisation() -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "route": {"type": "choice", "choice": "billing", "confidence": 0.88, "probabilities": {"billing": 0.88}}
            }
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
    )
    answer = result.answers["route"]
    assert answer.value == "billing"
    assert answer.confidence == 0.88
    assert answer.probabilities == {"billing": 0.88}


async def test_score_normalisation_derives_confidence_from_probabilities() -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "urgency": {
                    "type": "score",
                    "score": 2.4,
                    "probabilities": {"0": 0.01, "1": 0.04, "2": 0.49, "3": 0.46},
                }
            }
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2", "3"))],
    )
    answer = result.answers["urgency"]
    assert answer.value == 2.4
    assert answer.confidence == 0.49


async def test_multiple_questions_in_one_request() -> None:
    adapter, client = _adapter(_fixture("success"))
    result = await adapter.decide(
        OrchidDecisionState(text="charged twice"),
        [
            OrchidDecisionQuestion(name="relevant", kind="boolean"),
            OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
            OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2", "3")),
        ],
    )
    assert set(result.answers) == {"relevant", "route", "urgency"}
    assert client.payload is not None
    assert len(client.payload["questions"]) == 3


async def test_refusal_is_non_fatal_for_siblings() -> None:
    adapter, _ = _adapter(_fixture("refusal"))
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [
            OrchidDecisionQuestion(name="relevant", kind="boolean"),
            OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
        ],
    )
    assert result.answers["relevant"].value is True
    assert result.answers["route"].refusal == "insufficient context"
    assert result.answers["route"].value is None


async def test_required_missing_answer_raises() -> None:
    adapter, _ = _adapter(_fixture("partial"))
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_optional_missing_answer_is_omitted() -> None:
    adapter, _ = _adapter(_fixture("partial"))
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"), required=False)],
    )
    assert result.answers == {}


async def test_unknown_choice_option_raises() -> None:
    adapter, _ = _adapter({"answers": {"route": {"type": "choice", "choice": "sales"}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.5])
async def test_non_finite_or_out_of_range_boolean_raises(value: float) -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": value}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_malformed_answers_object_raises() -> None:
    adapter, _ = _adapter(_fixture("malformed"))
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_usage_is_normalised() -> None:
    adapter, _ = _adapter(_fixture("success"))
    result = await adapter.decide(
        OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="relevant", kind="boolean")]
    )
    assert result.usage == {"input_tokens": 120, "output_tokens": 0}
    assert result.provider == "fast"
    assert result.model == "decision-model-v1"
    assert result.latency_ms >= 0.0


async def test_timeout_is_forwarded() -> None:
    adapter, client = _adapter({"answers": {}})
    await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean", required=False)],
        timeout=0.5,
    )
    assert client.timeout == 0.5


async def test_transport_error_propagates_unchanged() -> None:
    error = OrchidDecisionError("provider timed out", code="timeout", retryable=True)
    adapter, _ = _adapter(error)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value is error


async def test_aclose_delegates() -> None:
    adapter, client = _adapter({"answers": {}})
    await adapter.aclose()
    assert client.closed is True
