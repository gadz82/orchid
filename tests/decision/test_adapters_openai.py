"""Tests for the OpenAI-compatible decision adapter."""

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
from orchid_ai.decision.openai_compatible import OpenAICompatibleDecisionModel

FIXTURES = Path(__file__).parent / "fixtures" / "openai_compatible.json"


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


def _adapter(response: Any) -> tuple[OpenAICompatibleDecisionModel, FakeClient]:
    client = FakeClient(response)
    return OpenAICompatibleDecisionModel(client=client, model="decision-model-v1", provider="fast"), client


def _fixture(name: str) -> dict[str, Any]:
    return json.loads(FIXTURES.read_text())[name]


async def test_request_shape_for_all_three_kinds() -> None:
    adapter, client = _adapter({"answers": []})
    questions = [
        OrchidDecisionQuestion(name="relevant", kind="boolean", instructions="Is it relevant?", required=False),
        OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"), required=False),
        OrchidDecisionQuestion(name="urgency", kind="score", rubric=("low", "medium", "high"), required=False),
    ]
    await adapter.decide(OrchidDecisionState(text="my invoice is wrong"), questions)

    assert client.path == "/v1/decisions"
    assert client.payload == {
        "model": "decision-model-v1",
        "input": "my invoice is wrong",
        "questions": [
            {"type": "predicate", "name": "relevant", "instructions": "Is it relevant?"},
            {
                "type": "choice",
                "name": "route",
                "instructions": "route",
                "choices": [{"value": "billing"}, {"value": "technical"}],
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "urgency",
                "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}],
            },
        ],
    }


async def test_state_with_payload_uses_a_user_message_with_input_json() -> None:
    adapter, client = _adapter({"answers": []})
    state = OrchidDecisionState(text="pick one", payload={"candidates": ["a", "b"]})
    await adapter.decide(state, [OrchidDecisionQuestion(name="q", kind="boolean", required=False)])
    assert client.payload is not None
    assert client.payload["input"] == [
        {
            "role": "user",
            "type": "message",
            "content": [
                {"type": "input_text", "text": "pick one"},
                {"type": "input_json", "data": {"candidates": ["a", "b"]}},
            ],
        }
    ]


async def test_boolean_normalisation() -> None:
    adapter, _ = _adapter({"answers": [{"type": "predicate", "name": "q", "probability": 0.3}]})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    answer = result.answers["q"]
    assert answer.value is False
    assert answer.confidence == pytest.approx(0.7)


async def test_choice_normalisation() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "billing",
                    "confidence": 0.88,
                    "probabilities": [
                        {"value": "billing", "probability": 0.88},
                        {"value": "technical", "probability": 0.12},
                    ],
                }
            ]
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
    )
    answer = result.answers["route"]
    assert answer.value == "billing"
    assert answer.confidence == 0.88
    assert answer.probabilities == {"billing": 0.88, "technical": 0.12}


async def test_score_normalisation_prefers_labels() -> None:
    adapter, _ = _adapter(_fixture("success"))
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [
            OrchidDecisionQuestion(name="relevant", kind="boolean"),
            OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
            OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2", "3")),
        ],
    )
    answer = result.answers["urgency"]
    assert answer.value == 2.4
    assert answer.confidence == 0.49
    assert answer.probabilities["2"] == 0.49


async def test_unnamed_answers_are_matched_by_position() -> None:
    adapter, _ = _adapter({"answers": [{"type": "predicate", "probability": 0.9}]})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is True


async def test_named_answers_may_arrive_out_of_order() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {"type": "predicate", "name": "b", "probability": 0.2},
                {"type": "predicate", "name": "a", "probability": 0.9},
            ]
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="a", kind="boolean"), OrchidDecisionQuestion(name="b", kind="boolean")],
    )
    assert result.answers["a"].value is True
    assert result.answers["b"].value is False


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
    assert result.answers["route"].refusal == "refused"


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


async def test_unknown_answer_name_is_rejected() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {"type": "predicate", "name": "q", "probability": 0.9},
                {"type": "predicate", "name": "extra", "probability": 0.9},
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_duplicate_answer_name_is_rejected() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {"type": "predicate", "name": "q", "probability": 0.9},
                {"type": "predicate", "name": "q", "probability": 0.1},
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_answer_type_mismatch_is_rejected() -> None:
    adapter, _ = _adapter({"answers": [{"type": "choice", "name": "q", "choice": "a", "confidence": 0.9}]})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_boolean_choice_value_is_rejected_for_string_options() -> None:
    adapter, _ = _adapter({"answers": [{"type": "choice", "name": "route", "choice": True, "confidence": 0.9}]})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("probabilities", [{"billing": 0.9}, "billing", [1, 2]])
async def test_invalid_probabilities_array_is_rejected(probabilities) -> None:
    adapter, _ = _adapter(
        {"answers": [{"type": "choice", "name": "route", "choice": "billing", "probabilities": probabilities}]}
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("probability", [-0.1, 1.5, math.nan, math.inf, 10**400])
async def test_invalid_probability_values_are_rejected(probability) -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "billing",
                    "probabilities": [{"value": "billing", "probability": probability}],
                }
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_huge_integer_score_is_rejected() -> None:
    adapter, _ = _adapter({"answers": [{"type": "score", "name": "urgency", "score": 10**400, "confidence": 0.9}]})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_usage_details_are_flattened() -> None:
    adapter, _ = _adapter(
        {
            "answers": [{"type": "predicate", "name": "relevant", "probability": 0.9}],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="relevant", kind="boolean", required=False)],
    )
    assert result.usage == {"input_tokens": 120, "output_tokens": 0, "cached_tokens": 10, "reasoning_tokens": 0}
    assert result.provider == "fast"
    assert result.model == "decision-model-v1"
    assert result.latency_ms >= 0.0


async def test_timeout_is_forwarded() -> None:
    adapter, client = _adapter({"answers": []})
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
    adapter, client = _adapter({"answers": []})
    await adapter.aclose()
    assert client.closed is True


async def test_unnamed_partial_answers_are_rejected() -> None:
    adapter, _ = _adapter({"answers": [{"type": "predicate", "probability": 0.9}]})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="a", kind="boolean"), OrchidDecisionQuestion(name="b", kind="boolean")],
        )
    assert excinfo.value.code == "invalid_response"


async def test_mixed_named_and_unnamed_out_of_position_is_rejected() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {"type": "predicate", "name": "c", "probability": 0.9},
                {"type": "predicate", "probability": 0.8},
                {"type": "predicate", "name": "a", "probability": 0.7},
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [
                OrchidDecisionQuestion(name="a", kind="boolean"),
                OrchidDecisionQuestion(name="b", kind="boolean"),
                OrchidDecisionQuestion(name="c", kind="boolean"),
            ],
        )
    assert excinfo.value.code == "invalid_response"


async def test_refusal_reason_is_preserved() -> None:
    adapter, _ = _adapter({"answers": [{"type": "refusal", "name": "q", "refusal": "insufficient context"}]})
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].refusal == "insufficient context"


async def test_bare_refusal_field_is_a_refusal() -> None:
    adapter, _ = _adapter(
        {"answers": [{"type": "predicate", "name": "q", "probability": 0.9, "refusal": "not applicable"}]}
    )
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].refusal == "not applicable"
    assert result.answers["q"].value is None


async def test_score_label_mismatch_falls_back_to_index() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {
                    "type": "score",
                    "name": "urgency",
                    "score": 1.2,
                    "probabilities": [
                        {"label": "low", "value": 0, "probability": 0.1},
                        {"label": "Medium", "value": 1, "probability": 0.9},
                    ],
                }
            ]
        }
    )
    result = await adapter.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="urgency", kind="score", rubric=("low", "mid", "high"))],
    )
    answer = result.answers["urgency"]
    assert answer.confidence == 0.9
    assert answer.probabilities == {"low": 0.1, "1": 0.9}


async def test_choice_label_outside_options_is_rejected() -> None:
    adapter, _ = _adapter({"answers": [{"type": "choice", "name": "route", "choice": "sales", "confidence": 0.9}]})
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -0.1, 1.5])
async def test_choice_and_score_confidence_is_validated(confidence: float) -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {"type": "choice", "name": "route", "choice": "billing", "confidence": confidence},
                {"type": "score", "name": "urgency", "score": 0.5, "confidence": confidence},
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [
                OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
                OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1")),
            ],
        )
    assert excinfo.value.code == "invalid_response"


async def test_explicit_null_confidence_is_invalid() -> None:
    adapter, _ = _adapter(
        {
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "billing",
                    "confidence": None,
                    "probabilities": [{"value": "billing", "probability": 0.9}],
                }
            ]
        }
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_usage_detail_cannot_override_top_level() -> None:
    adapter, _ = _adapter(
        {
            "answers": [{"type": "predicate", "name": "q", "probability": 0.9}],
            "usage": {"input_tokens": 100, "input_tokens_details": {"input_tokens": 1}},
        }
    )
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.usage["input_tokens"] == 100


@pytest.mark.parametrize(
    "probabilities",
    [
        [{"probability": 0.9}],
        [{"value": True, "probability": 0.9}],
        [{"value": "billing", "probability": math.nan}],
        [{"value": "billing", "probability": 0.9}, {"value": "billing", "probability": 0.1}],
    ],
)
async def test_malformed_choice_probability_entries_are_rejected(probabilities) -> None:
    adapter, _ = _adapter(
        {"answers": [{"type": "choice", "name": "route", "choice": "billing", "probabilities": probabilities}]}
    )
    with pytest.raises(OrchidDecisionError) as excinfo:
        await adapter.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_real_transport_round_trip() -> None:
    import httpx

    from orchid_ai.decision.http import DecisionHttpClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/decisions"
        body = json.loads(request.content)
        assert body["questions"][0]["type"] == "predicate"
        return httpx.Response(
            200,
            json={"answers": [{"type": "predicate", "name": "q", "probability": 0.95}]},
        )

    http_client = DecisionHttpClient(
        base_url="https://api.example.com",
        allowed_hosts=["api.example.com"],
        provider="fast",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=lambda host: ["203.0.113.10"],
    )
    adapter = OpenAICompatibleDecisionModel(client=http_client, model="decision-model-v1", provider="fast")
    result = await adapter.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is True
