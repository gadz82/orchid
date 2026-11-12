"""Tests for the decision contract in ``core/decision.py``."""

from __future__ import annotations

import math
from collections import UserDict

import pytest

from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    aggregate_shortlist,
    answers_or_none,
    require_confidence,
    validate_question_set,
    validate_threshold,
)


def _answer(name: str, value, confidence: float = 0.9, kind: str = "boolean", **kwargs):
    return OrchidDecisionAnswer(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        value=value,
        confidence=confidence,
        **kwargs,
    )


def _result(*answers: OrchidDecisionAnswer) -> OrchidDecisionResult:
    return OrchidDecisionResult(
        answers={answer.name: answer for answer in answers},
        provider="test",
        model="decision-model-v1",
        latency_ms=1.0,
    )


# ── State ────────────────────────────────────────────────────


def test_state_copies_payload() -> None:
    payload = {"candidates": ["a"]}
    state = OrchidDecisionState(text="hello", payload=payload)
    payload["candidates"].append("b")
    assert state.payload == {"candidates": ["a"]}


def test_state_rejects_non_string_text() -> None:
    with pytest.raises((TypeError, ValueError)):
        OrchidDecisionState(text=123)  # type: ignore[arg-type]


def test_state_rejects_non_mapping_payload() -> None:
    with pytest.raises((TypeError, ValueError)):
        OrchidDecisionState(payload=["not", "a", "mapping"])  # type: ignore[arg-type]


# ── Questions ────────────────────────────────────────────────


def test_boolean_question_rejects_options() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionQuestion(name="flag", kind="boolean", options=("a", "b"))


def test_choice_question_requires_two_options() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionQuestion(name="route", kind="choice", options=("only",))


def test_choice_question_rejects_duplicate_options() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionQuestion(name="route", kind="choice", options=("a", "a"))


def test_choice_question_accepts_options() -> None:
    question = OrchidDecisionQuestion(name="route", kind="choice", options=["billing", "technical"])
    assert question.options == ("billing", "technical")


def test_score_question_requires_rubric() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionQuestion(name="urgency", kind="score", rubric=("only",))


def test_score_question_accepts_rubric() -> None:
    question = OrchidDecisionQuestion(name="urgency", kind="score", rubric=("low", "high"))
    assert question.rubric == ("low", "high")


def test_question_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionQuestion(name="x", kind="ranking")  # type: ignore[arg-type]


def test_validate_question_set_rejects_duplicates() -> None:
    questions = [
        OrchidDecisionQuestion(name="flag", kind="boolean"),
        OrchidDecisionQuestion(name="flag", kind="boolean"),
    ]
    with pytest.raises(ValueError):
        validate_question_set(questions)


def test_validate_question_set_rejects_empty() -> None:
    with pytest.raises(ValueError):
        validate_question_set([])


# ── Answers ──────────────────────────────────────────────────


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf])
def test_answer_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        _answer("flag", True, confidence=confidence)


def test_answer_rejects_bool_for_score() -> None:
    with pytest.raises((TypeError, ValueError)):
        _answer("urgency", True, kind="score")


def test_answer_accepts_score_number() -> None:
    answer = _answer("urgency", 2.5, kind="score", probabilities={"1": 0.1, "2": 0.9})
    assert answer.value == 2.5
    assert answer.probabilities["2"] == 0.9


def test_answer_rejects_bool_value_for_boolean() -> None:
    with pytest.raises((TypeError, ValueError)):
        _answer("flag", "yes")


def test_refusal_carries_no_value() -> None:
    answer = OrchidDecisionAnswer(
        name="flag",
        kind="boolean",
        value=None,
        confidence=0.0,
        refusal="not applicable",
    )
    assert answer.refusal == "not applicable"
    assert answer.value is None


def test_refusal_rejects_value() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionAnswer(
            name="flag",
            kind="boolean",
            value=True,
            confidence=0.0,
            refusal="not applicable",
        )


# ── Results ──────────────────────────────────────────────────


def test_result_rejects_key_name_mismatch() -> None:
    answer = _answer("flag", True)
    with pytest.raises(ValueError):
        OrchidDecisionResult(answers={"other": answer}, provider="test", model="m")


def test_result_rejects_negative_latency() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionResult(answers={}, provider="test", model="m", latency_ms=-1.0)


def test_result_copies_answers() -> None:
    answer = _answer("flag", True)
    answers = {"flag": answer}
    result = OrchidDecisionResult(answers=answers, provider="test", model="m")
    answers.clear()
    assert "flag" in result.answers


# ── Errors ───────────────────────────────────────────────────


def test_error_carries_code_and_str() -> None:
    error = OrchidDecisionError("boom", code="timeout", provider="fast", retryable=True)
    assert error.code == "timeout"
    assert error.retryable is True
    assert str(error).startswith("timeout: boom")


def test_error_rejects_unknown_code() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionError("boom", code="exploded")  # type: ignore[arg-type]


# ── Helpers ──────────────────────────────────────────────────


@pytest.mark.parametrize("threshold", [-1, 2, math.nan])
def test_validate_threshold_rejects_invalid(threshold: float) -> None:
    with pytest.raises(ValueError):
        validate_threshold(threshold)


def test_require_confidence_returns_none_below_threshold() -> None:
    result = _result(_answer("flag", True, confidence=0.4))
    assert require_confidence(result, "flag", 0.8) is None


def test_require_confidence_returns_none_for_refusal() -> None:
    refusal = OrchidDecisionAnswer(name="flag", kind="boolean", value=None, confidence=0.9, refusal="nope")
    result = _result(refusal)
    assert require_confidence(result, "flag", 0.1) is None


def test_require_confidence_returns_answer_above_threshold() -> None:
    result = _result(_answer("flag", True, confidence=0.9))
    answer = require_confidence(result, "flag", 0.8)
    assert answer is not None and answer.value is True


def test_answers_or_none_is_all_or_nothing() -> None:
    result = _result(_answer("a", True), _answer("b", False, confidence=0.2))
    assert answers_or_none(result, ["a", "b"], threshold=0.5) is None
    selected = answers_or_none(result, ["a"], threshold=0.5)
    assert selected is not None and set(selected) == {"a"}


def test_aggregate_shortlist_selects_true_answers() -> None:
    result = _result(_answer("a", True), _answer("b", False), _answer("c", True, confidence=0.6))
    assert aggregate_shortlist(result, ["a", "b", "c"], threshold=0.5) == ["a", "c"]


def test_aggregate_shortlist_rejects_uncertain_inclusion() -> None:
    result = _result(_answer("a", True, confidence=0.9), _answer("b", True, confidence=0.4))
    assert aggregate_shortlist(result, ["a", "b"], threshold=0.5) is None


def test_aggregate_shortlist_ignores_unknown_and_non_boolean() -> None:
    result = _result(_answer("a", True), _answer("route", "billing", kind="choice"))
    assert aggregate_shortlist(result, ["a", "missing", "route"], threshold=0.5) == ["a"]


def test_aggregate_shortlist_returns_none_when_nothing_selected() -> None:
    result = _result(_answer("a", False))
    assert aggregate_shortlist(result, ["a"], threshold=0.5) is None


# ── Numeric and container hardening ─────────────────────────


def test_huge_integer_confidence_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        _answer("flag", True, confidence=10**400)


def test_huge_integer_score_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        _answer("urgency", 10**400, kind="score")


def test_huge_integer_latency_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionResult(answers={}, provider="test", model="m", latency_ms=10**400)


def test_huge_threshold_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        validate_threshold(10**400)


def test_answer_rejects_non_mapping_probabilities() -> None:
    with pytest.raises(TypeError):
        _answer("flag", True, probabilities=[("true", 0.9)])


def test_answer_rejects_non_finite_probability_value() -> None:
    with pytest.raises(ValueError):
        _answer("flag", True, probabilities={"true": math.nan})


def test_result_rejects_non_mapping_answers() -> None:
    with pytest.raises(TypeError):
        OrchidDecisionResult(answers=[], provider="test", model="m")  # type: ignore[arg-type]


def test_result_rejects_bool_usage() -> None:
    with pytest.raises(TypeError):
        OrchidDecisionResult(answers={}, provider="test", model="m", usage={"tokens": True})


def test_result_rejects_negative_usage() -> None:
    with pytest.raises(ValueError):
        OrchidDecisionResult(answers={}, provider="test", model="m", usage={"tokens": -1})


def test_question_rejects_bare_string_options() -> None:
    with pytest.raises(TypeError):
        OrchidDecisionQuestion(name="route", kind="choice", options="ab")


def test_question_rejects_non_bool_required() -> None:
    with pytest.raises(TypeError):
        OrchidDecisionQuestion(name="flag", kind="boolean", required="false")  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [{"x": math.nan}, {"x": {1, 2}}, {1: "x"}, {"x": b"bytes"}])
def test_state_rejects_non_json_safe_payload(payload: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        OrchidDecisionState(payload=payload)


def test_state_rejects_deep_payload() -> None:
    payload: dict = {}
    cursor = payload
    for _ in range(25):
        child: dict = {}
        cursor["n"] = child
        cursor = child
    with pytest.raises(ValueError):
        OrchidDecisionState(payload=payload)


def test_state_coerces_score_value_to_float() -> None:
    answer = _answer("urgency", 2, kind="score")
    assert isinstance(answer.value, float)


# ── Refusal and shortlist composition ───────────────────────


def test_refusal_does_not_break_sibling_answers() -> None:
    refusal = OrchidDecisionAnswer(name="b", kind="boolean", value=None, confidence=0.0, refusal="nope")
    result = _result(_answer("a", True), refusal)
    assert require_confidence(result, "a", 0.5) is not None
    assert require_confidence(result, "b", 0.5) is None


def test_answers_or_none_returns_none_for_missing_answer() -> None:
    result = _result(_answer("a", True))
    assert answers_or_none(result, ["a", "missing"], threshold=0.1) is None


def test_answers_or_none_returns_none_for_refusal() -> None:
    refusal = OrchidDecisionAnswer(name="b", kind="boolean", value=None, confidence=0.0, refusal="nope")
    result = _result(_answer("a", True), refusal)
    assert answers_or_none(result, ["a", "b"], threshold=0.1) is None


def test_answers_or_none_returns_none_for_empty_request() -> None:
    result = _result(_answer("a", True))
    assert answers_or_none(result, [], threshold=0.1) is None


def test_aggregate_shortlist_deduplicates_candidates() -> None:
    result = _result(_answer("a", True), _answer("b", True))
    assert aggregate_shortlist(result, ["a", "a", "b"], threshold=0.5) == ["a", "b"]


def test_aclose_defaults_to_noop() -> None:
    class _Provider(OrchidDecisionModel):
        async def decide(self, state, questions, *, timeout=None):
            raise NotImplementedError

    import asyncio

    assert asyncio.run(_Provider().aclose()) is None


def test_state_rejects_nested_non_dict_mapping() -> None:
    with pytest.raises(TypeError):
        OrchidDecisionState(payload={"nested": UserDict({"a": 1})})
