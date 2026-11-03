"""Adapter for the state-and-questions decision dialect.

Providers that speak this dialect accept a shared ``state`` and a map of named
questions, and return named answers with probabilities. The adapter owns the
wire mapping; the framework only sees :class:`OrchidDecisionResult`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    validate_question_set,
)
from .http import DecisionHttpClient

__all__ = ["StateQuestionsDecisionModel"]

_DECIDE_PATH = "/v1/decide"


class StateQuestionsDecisionModel(OrchidDecisionModel):
    """Decision model for the state-and-questions wire dialect.

    ``state`` is sent as a plain string when it carries no payload, or as
    ``{"text": ..., "context": ...}`` when structured context is present. A
    boolean question maps to the dialect's ``noul`` type; its value is the
    probability's side of ``0.5`` and its confidence is the probability of the
    selected side.
    """

    def __init__(
        self,
        *,
        client: DecisionHttpClient,
        model: str,
        provider: str = "",
        decide_path: str = _DECIDE_PATH,
    ) -> None:
        if not model.strip():
            raise ValueError("StateQuestionsDecisionModel requires a model id")
        self._client = client
        self._model = model
        self._provider = provider or model
        self._decide_path = decide_path

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Sequence[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        materialised = validate_question_set(questions)
        payload = {
            "model": self._model,
            "state": _serialise_state(state),
            "questions": {question.name: _question_payload(question) for question in materialised},
        }

        started = time.perf_counter()
        response = await self._client.request_json(self._decide_path, payload, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw_answers = response.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise self._invalid("response is missing an answers object")

        answers: dict[str, OrchidDecisionAnswer] = {}
        for question in materialised:
            raw = raw_answers.get(question.name)
            if raw is None:
                if question.required:
                    raise self._invalid(f"response is missing required answer {question.name!r}")
                continue
            answers[question.name] = _normalise_answer(question, raw)

        return OrchidDecisionResult(
            answers=answers,
            provider=self._provider,
            model=self._model,
            latency_ms=latency_ms,
            usage=_normalise_usage(response.get("usage")),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _invalid(self, message: str) -> OrchidDecisionError:
        return OrchidDecisionError(
            message,
            code="invalid_response",
            provider=self._provider,
            retryable=False,
            host=self._client.host,
        )


def _serialise_state(state: OrchidDecisionState) -> Any:
    if state.payload:
        return {"text": state.text, "context": dict(state.payload)}
    return state.text


def _question_payload(question: OrchidDecisionQuestion) -> dict[str, Any]:
    if question.kind == "boolean":
        payload: dict[str, Any] = {"type": "noul"}
    elif question.kind == "choice":
        payload = {"type": "choice", "criteria": {option: option for option in question.options}}
    else:
        payload = {"type": "score", "criteria": list(question.rubric)}
    if question.instructions:
        payload["instructions"] = question.instructions
    return payload


def _normalise_answer(question: OrchidDecisionQuestion, raw: Any) -> OrchidDecisionAnswer:
    if not isinstance(raw, Mapping):
        raise _invalid_answer(question, "answer must be an object")

    answer_type = raw.get("type")
    if answer_type == "refusal" or raw.get("refusal") is not None:
        reason = raw.get("refusal") or raw.get("reason") or "refused"
        return OrchidDecisionAnswer(
            name=question.name,
            kind=question.kind,
            value=None,
            confidence=0.0,
            refusal=str(reason),
        )

    if question.kind == "boolean":
        probability = _probability(raw)
        if probability is None:
            raise _invalid_answer(question, "boolean answer is missing a probability")
        value = probability >= 0.5
        confidence = probability if value else 1.0 - probability
        return OrchidDecisionAnswer(
            name=question.name,
            kind="boolean",
            value=value,
            confidence=confidence,
            probabilities={"true": probability, "false": 1.0 - probability},
        )

    if question.kind == "choice":
        selected = raw.get("choice")
        if not isinstance(selected, str) or selected not in question.options:
            raise _invalid_answer(question, f"choice answer {selected!r} is not a declared option")
        confidence = _confidence(raw)
        probabilities = _probabilities(raw, expected=set(question.options))
        return OrchidDecisionAnswer(
            name=question.name,
            kind="choice",
            value=selected,
            confidence=confidence if confidence is not None else probabilities.get(selected, 1.0),
            probabilities=probabilities,
        )

    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise _invalid_answer(question, "score answer is missing a numeric value")
    if not math.isfinite(float(score)):
        raise _invalid_answer(question, "score answer must be finite")
    probabilities = _probabilities(raw, expected=set(question.rubric) | {str(i) for i in range(len(question.rubric))})
    confidence = _confidence(raw)
    if confidence is None:
        confidence = max(probabilities.values(), default=1.0)
    return OrchidDecisionAnswer(
        name=question.name,
        kind="score",
        value=float(score),
        confidence=confidence,
        probabilities=probabilities,
    )


def _probability(raw: Mapping[str, Any]) -> float | None:
    for key in ("noul", "probability", "confidence"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            return number
    return None


def _confidence(raw: Mapping[str, Any]) -> float | None:
    value = raw.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _probabilities(raw: Mapping[str, Any], *, expected: set[str]) -> dict[str, float]:
    raw_probabilities = raw.get("probabilities")
    if not isinstance(raw_probabilities, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in raw_probabilities.items():
        if not isinstance(key, str) or key not in expected:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            result[key] = number
    return result


def _normalise_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(key, str) and isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            usage[key] = raw
    return usage


def _invalid_answer(question: OrchidDecisionQuestion, message: str) -> OrchidDecisionError:
    return OrchidDecisionError(
        f"invalid answer for {question.name!r}: {message}",
        code="invalid_response",
        retryable=False,
    )
