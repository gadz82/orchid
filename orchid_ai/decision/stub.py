"""Deterministic stub decision provider.

The stub keeps tests and the example's offline profile runnable without any
provider key. It answers every question kind deterministically and supports
explicit low-confidence and failure modes so fallback paths can be exercised.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    validate_question_set,
)
from .answers import normalise_boolean, normalise_choice, normalise_score, refusal_answer

__all__ = ["StubDecisionModel"]

StubMode = Literal["answer", "low_confidence", "fail"]


class StubDecisionModel(OrchidDecisionModel):
    """Deterministic provider for tests and the offline example.

    # ORCHID-MEM: 01a1a186-107d-74d2-9921-7f329b0b019e — Deterministic stub decision provider for offline runs and fallback tests.

    ``answers`` maps a question name (or ``"<state text>:<name>"`` for a
    context-specific answer) to a spec with ``value``/``probability``,
    ``choice``, ``score``, ``confidence``, or ``refusal``.
    """

    def __init__(
        self,
        *,
        answers: Mapping[str, Mapping[str, Any]] | None = None,
        confidence: float = 0.9,
        mode: StubMode = "answer",
        provider: str = "stub",
        model: str = "stub",
    ) -> None:
        if mode not in ("answer", "low_confidence", "fail"):
            raise ValueError(f"unknown stub mode {mode!r}")
        self._answers = answers or {}
        self._confidence = 0.55 if mode == "low_confidence" else max(0.0, min(1.0, confidence))
        self._mode = mode
        self._provider = provider
        self._model = model

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Sequence[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        materialised = validate_question_set(questions)
        if self._mode == "fail":
            raise OrchidDecisionError(
                "stub provider is configured to fail",
                code="provider_unavailable",
                provider=self._provider,
                retryable=False,
            )

        started = time.perf_counter()
        answers: dict[str, OrchidDecisionAnswer] = {}
        for question in materialised:
            spec = self._spec(state, question)
            answers[question.name] = self._answer(question, spec)

        return OrchidDecisionResult(
            answers=answers,
            provider=self._provider,
            model=self._model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _spec(self, state: OrchidDecisionState, question: OrchidDecisionQuestion) -> Mapping[str, Any]:
        contextual = f"{state.text}:{question.name}"
        if contextual in self._answers:
            return self._answers[contextual]
        return self._answers.get(question.name, {})

    def _answer(self, question: OrchidDecisionQuestion, spec: Mapping[str, Any]) -> OrchidDecisionAnswer:
        def error(message: str) -> OrchidDecisionError:
            return OrchidDecisionError(
                f"invalid stub answer for {question.name!r}: {message}",
                code="invalid_response",
                provider=self._provider,
                retryable=False,
            )

        if spec.get("refusal"):
            return refusal_answer(question, spec["refusal"])

        confidence = float(spec.get("confidence", self._confidence))
        if question.kind == "boolean":
            if "probability" in spec:
                probability = spec["probability"]
            else:
                probability = confidence if spec.get("value", True) else 1.0 - confidence
            return normalise_boolean(question, probability, error)
        if question.kind == "choice":
            return normalise_choice(question, spec.get("choice", question.options[0]), confidence, {}, error)
        return normalise_score(question, spec.get("score", 0.0), confidence, {}, error)
