"""Adapter for the state-and-questions decision dialect.

Providers that speak this dialect accept a shared ``state`` and a map of named
questions, and return named answers with probabilities. The adapter owns the
wire mapping; the framework only sees :class:`OrchidDecisionResult`.

Wire shape::

    {
      "model": "...",
      "state": "<text>" | {"text": "...", "context": {...}},
      "questions": {"name": {"type": "noul" | "choice" | "score", ...}}
    }

``state`` is a bare string when the state carries no structured payload; it is
an object with ``text`` and ``context`` when a payload is present, so a payload
key named ``text`` cannot collide with the state text.

Malformed question sets (duplicate names, an empty list, a non-question
element) are programmer errors and raise ``ValueError``/``TypeError``; provider
responses are untrusted and always fail as :class:`OrchidDecisionError`.
"""

from __future__ import annotations

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
from .answers import (
    ErrorFactory,
    check_answer_type,
    finite_number,
    normalise_boolean,
    normalise_choice,
    normalise_score,
    normalise_usage,
    refusal_answer,
)
from .http import DecisionHttpClient

__all__ = ["StateQuestionsDecisionModel"]

_DECIDE_PATH = "/v1/decide"
_EXPECTED_TYPE = {"boolean": "noul", "choice": "choice", "score": "score"}
_USAGE_KEYS = frozenset({"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"})


class StateQuestionsDecisionModel(OrchidDecisionModel):
    """Decision model for the state-and-questions wire dialect.

    # ORCHID-MEM: 01a1a158-10f2-7505-9f53-733ce9d46e42 — State-and-questions decision adapter with strict answer validation.

    A boolean question maps to the dialect's ``noul`` type; its value is the
    probability's side of ``0.5`` and its confidence is the probability of the
    selected side (so a confident ``False`` reports the probability of false).
    Choice and score answers must carry confidence evidence: a ``confidence``
    field or a declared probability for the selected value. Missing evidence is
    an ``invalid_response``, never a fabricated ``1.0``.
    """

    def __init__(
        self,
        *,
        client: DecisionHttpClient,
        model: str,
        provider: str,
        decide_path: str = _DECIDE_PATH,
    ) -> None:
        if not model.strip():
            raise ValueError("StateQuestionsDecisionModel requires a model id")
        if not provider.strip():
            raise ValueError("StateQuestionsDecisionModel requires a provider key")
        self._client = client
        self._model = model
        self._provider = provider
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

        expected = {question.name for question in materialised}
        for name in raw_answers:
            if name not in expected:
                raise self._invalid("response contains an answer that was not requested")

        answers: dict[str, OrchidDecisionAnswer] = {}
        for question in materialised:
            raw = raw_answers.get(question.name)
            if raw is None:
                if question.required:
                    raise self._invalid(f"response is missing required answer {question.name!r}")
                continue
            answers[question.name] = _normalise_answer(question, raw, self._error_for(question))

        return OrchidDecisionResult(
            answers=answers,
            provider=self._provider,
            model=self._model,
            latency_ms=latency_ms,
            usage=normalise_usage(response.get("usage"), keys=_USAGE_KEYS),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _error_for(self, question: OrchidDecisionQuestion) -> ErrorFactory:
        def error(message: str) -> OrchidDecisionError:
            return self._invalid(f"invalid answer for {question.name!r}: {message}")

        return error

    def _invalid(self, message: str) -> OrchidDecisionError:
        return OrchidDecisionError(
            message,
            code="invalid_response",
            provider=self._provider,
            retryable=False,
            host=getattr(self._client, "host", ""),
        )


def _serialise_state(state: OrchidDecisionState) -> Any:
    if state.payload:
        return {"text": state.text, "context": dict(state.payload)}
    return state.text


def _question_payload(question: OrchidDecisionQuestion) -> dict[str, Any]:
    if question.kind == "boolean":
        payload: dict[str, Any] = {"type": "noul"}
    elif question.kind == "choice":
        # The dialect expects a label -> meaning map; without per-option
        # descriptions the option is its own meaning.
        payload = {"type": "choice", "criteria": {option: option for option in question.options}}
    else:
        payload = {"type": "score", "criteria": list(question.rubric)}
    if question.instructions:
        payload["instructions"] = question.instructions
    return payload


def _normalise_answer(question: OrchidDecisionQuestion, raw: Any, error: ErrorFactory) -> OrchidDecisionAnswer:
    if not isinstance(raw, Mapping):
        raise error("answer must be an object")

    answer_type = raw.get("type")
    check_answer_type(question, answer_type, _EXPECTED_TYPE[question.kind], error)

    refusal = raw.get("refusal")
    if answer_type == "refusal" or (isinstance(refusal, str) and refusal.strip()):
        reason = refusal if isinstance(refusal, str) and refusal.strip() else raw.get("reason")
        return refusal_answer(question, reason)
    if refusal is not None and not isinstance(refusal, str):
        raise error("refusal must be a string")

    if question.kind == "boolean":
        return normalise_boolean(question, _boolean_probability(raw, error), error)
    if question.kind == "choice":
        return normalise_choice(
            question,
            raw.get("choice"),
            raw.get("confidence"),
            raw.get("probabilities"),
            error,
        )
    return normalise_score(
        question,
        raw.get("score"),
        raw.get("confidence"),
        raw.get("probabilities"),
        error,
    )


def _boolean_probability(raw: Mapping[str, Any], error: ErrorFactory) -> float:
    for key in ("noul", "probability"):
        if key not in raw:
            continue
        probability = finite_number(raw[key])
        if probability is None or not 0.0 <= probability <= 1.0:
            raise error(f"{key} must be finite and within [0, 1]")
        return probability
    raise error("boolean answer is missing a probability")
