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
    assert client.payload == {
        "model": "decision-model-v1",
        "state": "my invoice is wrong",
        "questions": {
            "relevant": {"type": "noul", "instructions": "Is it relevant?"},
            "route": {"type": "choice", "criteria": {"billing": "billing", "technical": "technical"}},
            "urgency": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
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
    adapter, _ = _adapter(
        {"answers": {"relevant": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 120, "output_tokens": 0}}
    )
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


# ── Huge numbers and confidence evidence ───────────────────


async def test_huge_integer_score_is_invalid_response() -> None:
    adapter, _ = _adapter({"answers": {"urgency": {"type": "score", "score": 10**400, "confidence": 0.9}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_boolean_missing_probability_is_invalid() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "confidence": 0.9}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_boolean_invalid_noul_is_not_rescued_by_confidence() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 1.5, "confidence": 0.9}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -0.1, 1.5, "0.9", True, 10**400])
async def test_choice_invalid_confidence_is_invalid(confidence) -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": confidence,
                    "probabilities": {"billing": 0.9},
                }
            }
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -0.1, 1.5, "0.9", True, 10**400])
async def test_score_invalid_confidence_is_invalid(confidence) -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "urgency": {
                    "type": "score",
                    "score": 0.5,
                    "confidence": confidence,
                    "probabilities": {"0": 0.5, "1": 0.5},
                }
            }
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_choice_without_confidence_evidence_is_invalid() -> None:
    adapter, _ = _adapter({"answers": {"route": {"type": "choice", "choice": "billing"}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_choice_confidence_can_come_from_selected_probability() -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.77, "technical": 0.23}}
            }
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
    )
    assert result.answers["route"].confidence == 0.77


async def test_score_without_confidence_evidence_is_invalid() -> None:
    adapter, _ = _adapter({"answers": {"urgency": {"type": "score", "score": 0.5}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf, "0.5", 10**400])
async def test_declared_probability_values_are_validated(probability) -> None:
    adapter, _ = _adapter(
        {"answers": {"route": {"type": "choice", "choice": "billing", "probabilities": {"billing": probability}}}}
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


# ── Type safety and refusals ───────────────────────────────


async def test_answer_type_mismatch_is_invalid() -> None:
    adapter, _ = _adapter({"answers": {"route": {"type": "noul", "noul": 0.9, "choice": "billing"}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_falsy_non_string_refusal_is_invalid() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 0.9, "refusal": 0}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_empty_refusal_string_is_not_a_refusal() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 0.9, "refusal": ""}}})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is True
    assert result.answers["q"].refusal is None


async def test_refusal_reason_is_sanitised_and_capped() -> None:
    reason = ("bad\x00\x07 reason " * 200).strip()
    adapter, _ = _adapter({"answers": {"q": {"type": "refusal", "reason": reason}}})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert len(result.answers["q"].refusal or "") <= 500
    assert "\x00" not in (result.answers["q"].refusal or "")


async def test_error_does_not_echo_provider_choice_value() -> None:
    sentinel = "STATE-ECHO-abcdef"
    adapter, _ = _adapter({"answers": {"route": {"type": "choice", "choice": sentinel}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert sentinel not in str(excinfo.value)


async def test_error_carries_provider_and_host() -> None:
    adapter, _ = _adapter({"answers": {"route": {"type": "choice", "choice": "sales"}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.provider == "fast"
    assert excinfo.value.host == "api.example.com"
    assert excinfo.value.retryable is False


# ── Unknown answers, score range, usage, payload collision ──


async def test_unknown_answer_name_is_rejected() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 0.9}, "extra": {"type": "noul", "noul": 0.9}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("score", [-0.1, 2.1, 100.0])
async def test_score_outside_rubric_range_is_invalid(score: float) -> None:
    adapter, _ = _adapter({"answers": {"urgency": {"type": "score", "score": score, "confidence": 0.9}}})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_usage_unknown_keys_are_dropped() -> None:
    adapter, _ = _adapter({"answers": {}, "usage": {"input_tokens": 5, "secret": 99, "output_tokens": -1}})
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean", required=False)],
    )
    assert result.usage == {"input_tokens": 5}


async def test_payload_keys_text_and_context_do_not_collide() -> None:
    adapter, client = _adapter({"answers": {}})
    state = OrchidDecisionState(text="outer", payload={"text": "inner", "context": "value"})
    await adapter.decide(state, [OrchidDecisionQuestion(name="q", kind="boolean", required=False)])
    assert client.payload is not None
    assert client.payload["state"] == {
        "text": "outer",
        "context": {"text": "inner", "context": "value"},
    }


async def test_boolean_tie_resolves_to_true_with_half_confidence() -> None:
    adapter, _ = _adapter({"answers": {"q": {"type": "noul", "noul": 0.5}}})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is True
    assert result.answers["q"].confidence == 0.5


async def test_real_transport_huge_integer_is_invalid_response() -> None:
    import httpx

    from orchid_ai.decision.http import DecisionHttpClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answers": {"urgency": {"type": "score", "score": 10**400, "confidence": 0.9}}},
        )

    http_client = DecisionHttpClient(
        base_url="https://api.example.com",
        allowed_hosts=["api.example.com"],
        provider="fast",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=lambda host: ["203.0.113.10"],
    )
    adapter = StateQuestionsDecisionModel(client=http_client, model="decision-model-v1", provider="fast")
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_explicit_null_choice_confidence_is_invalid() -> None:
    adapter, _ = _adapter(
        {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": None,
                    "probabilities": {"billing": 0.9},
                }
            }
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_explicit_null_probabilities_is_invalid() -> None:
    adapter, _ = _adapter(
        {"answers": {"route": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": None}}}
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"
