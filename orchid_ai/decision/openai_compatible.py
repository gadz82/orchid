"""Adapter for the OpenAI-compatible decisions dialect.

The dialect takes a shared ``input`` and an ordered array of questions
(``predicate``, ``choice``, ``score``) and returns an ordered array of answers.
Names are optional on the wire, so an unnamed answer is matched to the question
at the same position.

Wire shape::

    {
      "model": "...",
      "input": "<text>" | [{"role": "user", "content": [{"type": "input_text", ...}, {"type": "input_json", ...}]}],
      "questions": [{"type": "predicate" | "choice" | "score", "name": "...", ...}]
    }
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
    normalise_boolean,
    normalise_choice,
    normalise_score,
    normalise_usage,
    refusal_answer,
)
from .http import DecisionHttpClient

__all__ = ["OpenAICompatibleDecisionModel"]

_DECIDE_PATH = "/v1/decisions"
_EXPECTED_TYPE = {"boolean": "predicate", "choice": "choice", "score": "score"}
_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "cache_write_tokens", "reasoning_tokens"}
)


class OpenAICompatibleDecisionModel(OrchidDecisionModel):
    """Decision model for the OpenAI-compatible decisions dialect.

    # ORCHID-MEM: 01a1a15a-5156-738a-86fb-e879b7333067 — OpenAI-compatible decision adapter with positional answer matching.

    Boolean questions use the ``predicate`` type. Answers are returned in
    question order with optional names; an unnamed answer is matched by
    position. Confidence evidence is validated by the shared answer helpers, so
    this adapter and the state-and-questions adapter produce equivalent results
    for equivalent inputs.
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
            raise ValueError("OpenAICompatibleDecisionModel requires a model id")
        if not provider.strip():
            raise ValueError("OpenAICompatibleDecisionModel requires a provider key")
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
            "input": _serialise_input(state),
            "questions": [_question_payload(question) for question in materialised],
        }

        started = time.perf_counter()
        response = await self._client.request_json(self._decide_path, payload, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw_answers = response.get("answers")
        if not isinstance(raw_answers, list):
            raise self._invalid("response is missing an answers array")
        indexed = self._index_answers(raw_answers, materialised)

        answers: dict[str, OrchidDecisionAnswer] = {}
        for question in materialised:
            raw = indexed.get(question.name)
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
            usage=_flatten_usage(response.get("usage")),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _index_answers(
        self,
        raw_answers: list[Any],
        questions: Sequence[OrchidDecisionQuestion],
    ) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for position, raw in enumerate(raw_answers):
            if not isinstance(raw, Mapping):
                raise self._invalid("answer entries must be objects")
            name = raw.get("name")
            if name is None or name == "":
                if position >= len(questions):
                    raise self._invalid("unnamed answer has no matching question")
                name = questions[position].name
            if not isinstance(name, str):
                raise self._invalid("answer name must be a string")
            if name not in {question.name for question in questions}:
                raise self._invalid("response contains an answer that was not requested")
            if name in indexed:
                raise self._invalid("response contains a duplicate answer name")
            indexed[name] = raw
        return indexed

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


def _serialise_input(state: OrchidDecisionState) -> Any:
    if not state.payload:
        return state.text
    return [
        {
            "role": "user",
            "type": "message",
            "content": [
                {"type": "input_text", "text": state.text},
                {"type": "input_json", "data": dict(state.payload)},
            ],
        }
    ]


def _question_payload(question: OrchidDecisionQuestion) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": _EXPECTED_TYPE[question.kind],
        "name": question.name,
        "instructions": question.instructions or question.name,
    }
    if question.kind == "choice":
        payload["choices"] = [{"value": option} for option in question.options]
    elif question.kind == "score":
        payload["levels"] = [{"label": level} for level in question.rubric]
    return payload


def _normalise_answer(question: OrchidDecisionQuestion, raw: Any, error: ErrorFactory) -> OrchidDecisionAnswer:
    if not isinstance(raw, Mapping):
        raise error("answer must be an object")

    answer_type = raw.get("type")
    check_answer_type(question, answer_type, _EXPECTED_TYPE[question.kind], error)

    if answer_type == "refusal":
        return refusal_answer(question, "refused")

    if question.kind == "boolean":
        return normalise_boolean(question, raw.get("probability"), error)
    if question.kind == "choice":
        return normalise_choice(
            question,
            raw.get("choice"),
            raw.get("confidence"),
            _choice_probabilities(raw.get("probabilities"), error),
            error,
        )
    return normalise_score(
        question,
        raw.get("score"),
        raw.get("confidence"),
        _score_probabilities(raw.get("probabilities"), error),
        error,
    )


def _choice_probabilities(value: Any, error: ErrorFactory) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise error("probabilities must be an array")
    result: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, Mapping) or "value" not in entry:
            raise error("probability entries must carry a value")
        key = entry["value"]
        if not isinstance(key, str):
            continue
        result[key] = entry.get("probability")
    return result


def _score_probabilities(value: Any, error: ErrorFactory) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise error("probabilities must be an array")
    result: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, Mapping):
            raise error("probability entries must be objects")
        label = entry.get("label")
        if isinstance(label, str):
            result[label] = entry.get("probability")
            continue
        index = entry.get("value")
        if isinstance(index, int) and not isinstance(index, bool):
            result[str(index)] = entry.get("probability")
    return result


def _flatten_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    flat: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in value:
            flat[key] = value[key]
    for detail_key in ("input_tokens_details", "output_tokens_details"):
        detail = value.get(detail_key)
        if isinstance(detail, Mapping):
            flat.update(detail)
    return normalise_usage(flat, keys=_USAGE_KEYS)
