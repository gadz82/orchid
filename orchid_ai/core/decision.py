"""Decision-model abstractions — typed judgements instead of generated text.

A decision model evaluates a *state* against named *questions* and returns
typed answers with calibrated confidence (boolean, choice, or score). It does
not generate prose. Providers such as System One decision models and the
OpenAI Decisions API speak different wire formats, but every consumer in the
framework depends on the single contract defined here.

This module is deliberately stdlib-only: transport, provider dialects, and
configuration live in :mod:`orchid_ai.decision` and ``config/``.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "DecisionErrorCode",
    "DecisionKind",
    "OrchidDecisionAnswer",
    "OrchidDecisionError",
    "OrchidDecisionModel",
    "OrchidDecisionQuestion",
    "OrchidDecisionResult",
    "OrchidDecisionState",
    "aggregate_shortlist",
    "answers_or_none",
    "require_confidence",
    "validate_question_set",
    "validate_threshold",
]

DecisionKind = Literal["boolean", "choice", "score"]

DecisionErrorCode = Literal[
    "not_configured",
    "timeout",
    "provider_unavailable",
    "egress_denied",
    "invalid_response",
    "refusal",
    "budget_exhausted",
]

_DECISION_KINDS: frozenset[str] = frozenset({"boolean", "choice", "score"})
_DECISION_ERROR_CODES: frozenset[str] = frozenset(
    {
        "not_configured",
        "timeout",
        "provider_unavailable",
        "egress_denied",
        "invalid_response",
        "refusal",
        "budget_exhausted",
    }
)


def _validate_threshold_value(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"threshold must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"threshold must be finite and within [0, 1], got {value!r}")
    return number


def _validate_confidence_value(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1], got {value!r}")
    return number


def _validate_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    copied: dict[str, float] = {}
    for key, value in probabilities.items():
        if not isinstance(key, str):
            raise TypeError("probability keys must be strings")
        copied[key] = _validate_confidence_value(value, field_name=f"probability[{key!r}]")
    return copied


def _validate_kind(kind: str) -> None:
    if kind not in _DECISION_KINDS:
        raise ValueError(f"kind must be one of {sorted(_DECISION_KINDS)}, got {kind!r}")


@dataclass(frozen=True)
class OrchidDecisionState:
    """The context a decision model evaluates.

    ``payload`` carries structured context (candidates, passages, counts). It
    is copied on construction so a caller cannot mutate the state after the
    decision starts, and it must stay JSON-safe because transports serialise it.
    """

    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("state text must be a string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("state payload must be a mapping")
        object.__setattr__(self, "payload", copy.deepcopy(dict(self.payload)))


@dataclass(frozen=True)
class OrchidDecisionQuestion:
    """One named judgement inside a decision request.

    ``boolean`` questions have no options. ``choice`` questions carry at least
    two options. ``score`` questions carry at least two ordered rubric levels,
    low to high.
    """

    name: str
    kind: DecisionKind
    instructions: str = ""
    options: tuple[str, ...] = ()
    rubric: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("question name must be a non-empty string")
        if not isinstance(self.instructions, str):
            raise TypeError("question instructions must be a string")

        options = tuple(self.options)
        rubric = tuple(self.rubric)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "rubric", rubric)

        if self.kind == "boolean":
            if options or rubric:
                raise ValueError("boolean questions do not accept options or rubric")
            return

        if self.kind == "choice":
            _validate_label_set(options, field_name="options", minimum=2)
            if rubric:
                raise ValueError("choice questions do not accept a rubric")
            return

        _validate_label_set(rubric, field_name="rubric", minimum=2)
        if options:
            raise ValueError("score questions do not accept options")


def _validate_label_set(values: tuple[str, ...], *, field_name: str, minimum: int) -> None:
    if len(values) < minimum:
        raise ValueError(f"{field_name} requires at least {minimum} entries")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


@dataclass(frozen=True)
class OrchidDecisionAnswer:
    """The normalised result for one question.

    A refusal is a valid answer: ``value`` is ``None``, ``refusal`` explains
    why, and sibling answers remain usable. ``confidence`` is always finite and
    within ``[0, 1]`` so a caller can compare it with a threshold without
    defensive checks.
    """

    name: str
    kind: DecisionKind
    value: bool | str | float | None
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    refusal: str | None = None

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("answer name must be a non-empty string")

        object.__setattr__(
            self,
            "confidence",
            _validate_confidence_value(self.confidence, field_name="confidence"),
        )
        object.__setattr__(self, "probabilities", _validate_probabilities(self.probabilities))

        if self.refusal is not None:
            if not isinstance(self.refusal, str):
                raise ValueError("refusal must be a string")
            if self.value is not None:
                raise ValueError("a refused answer must not carry a value")
            return

        if self.kind == "boolean":
            if not isinstance(self.value, bool):
                raise ValueError("boolean answers require a bool value")
            return

        if self.kind == "choice":
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("choice answers require a non-empty string value")
            return

        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("score answers require a numeric value")
        if not math.isfinite(float(self.value)):
            raise ValueError("score answers require a finite value")


@dataclass(frozen=True)
class OrchidDecisionResult:
    """Normalised answers for one decision request."""

    answers: Mapping[str, OrchidDecisionAnswer]
    provider: str
    model: str
    latency_ms: float = 0.0
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("result provider must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("result model must be a non-empty string")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise TypeError("latency_ms must be a number")
        if not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")

        copied: dict[str, OrchidDecisionAnswer] = {}
        for key, answer in self.answers.items():
            if not isinstance(key, str) or not isinstance(answer, OrchidDecisionAnswer):
                raise TypeError("answers must map string names to OrchidDecisionAnswer values")
            if key != answer.name:
                raise ValueError(f"answer key {key!r} does not match answer name {answer.name!r}")
            copied[key] = answer
        object.__setattr__(self, "answers", copied)
        object.__setattr__(self, "usage", {str(k): int(v) for k, v in self.usage.items()})


class OrchidDecisionModel(ABC):
    """Provider-agnostic contract for decision models.

    # ORCHID-MEM: 01a1a102-9e83-7b0a-8a8e-333d3a323039 — Provider-agnostic typed decision contract.

    Implementations translate :class:`OrchidDecisionQuestion` values into their
    provider's wire shape and normalise the response. They must not perform
    side effects, compose tool arguments, or decide workflow on their own: they
    answer questions, and the framework owns the branch.

    Failure modes:
      Implementations raise :class:`OrchidDecisionError` with a stable code.
      Callers treat any error as "no decision" and run their declared fallback;
      they never retry inside a request beyond the configured budget.
    """

    @abstractmethod
    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Sequence[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        """Answer ``questions`` about ``state``.

        ``timeout`` is a per-call ceiling in seconds; ``None`` uses the
        provider's configured default. Implementations must validate the
        response (finite confidence, declared option values, required answers)
        before returning.
        """
        ...


class OrchidDecisionError(Exception):
    """A decision call failed with a stable, non-leaking error code.

    The message, code, status, and host are safe to log and to surface in
    metadata. Response bodies and state never appear here.
    """

    def __init__(
        self,
        message: str,
        *,
        code: DecisionErrorCode,
        provider: str = "",
        retryable: bool = False,
        status_code: int | None = None,
        host: str = "",
    ) -> None:
        if code not in _DECISION_ERROR_CODES:
            raise ValueError(f"unknown decision error code {code!r}")
        super().__init__(message)
        self.code: str = code
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code
        self.host = host

    def __str__(self) -> str:
        return f"{self.code}: {super().__str__()}"


def validate_threshold(threshold: float) -> float:
    """Return ``threshold`` as a float after validating the confidence range."""
    return _validate_threshold_value(threshold)


def validate_question_set(questions: Sequence[OrchidDecisionQuestion]) -> tuple[OrchidDecisionQuestion, ...]:
    """Validate a request's questions and return them as a tuple.

    Names must be unique because answers are keyed by name. An empty request is
    rejected: a decision without a question cannot inform a branch.
    """
    materialised = tuple(questions)
    if not materialised:
        raise ValueError("a decision request requires at least one question")
    seen: set[str] = set()
    for question in materialised:
        if not isinstance(question, OrchidDecisionQuestion):
            raise TypeError("questions must be OrchidDecisionQuestion values")
        if question.name in seen:
            raise ValueError(f"duplicate question name {question.name!r}")
        seen.add(question.name)
    return materialised


def require_confidence(
    result: OrchidDecisionResult,
    name: str,
    threshold: float,
) -> OrchidDecisionAnswer | None:
    """Return the answer for ``name`` when it is usable, else ``None``.

    ``None`` means "no confident decision": the answer is missing, refused, or
    below ``threshold``. Callers must branch on ``None`` rather than treating a
    missing boolean as ``False``.
    """
    limit = _validate_threshold_value(threshold)
    answer = result.answers.get(name)
    if answer is None or answer.refusal is not None:
        return None
    if answer.confidence < limit:
        return None
    return answer


def answers_or_none(
    result: OrchidDecisionResult,
    names: Sequence[str],
    *,
    threshold: float = 0.0,
) -> dict[str, OrchidDecisionAnswer] | None:
    """Return every named answer when all are usable, else ``None``.

    All-or-nothing keeps a partial response from being mistaken for a complete
    decision.
    """
    limit = _validate_threshold_value(threshold)
    selected: dict[str, OrchidDecisionAnswer] = {}
    for name in names:
        answer = require_confidence(result, name, limit)
        if answer is None:
            return None
        selected[name] = answer
    return selected


def aggregate_shortlist(
    result: OrchidDecisionResult,
    candidates: Sequence[str],
    *,
    threshold: float = 0.5,
) -> list[str] | None:
    """Return the candidates a boolean decision selected, or ``None``.

    Each candidate is a separate boolean question. Only ``True`` answers are
    considered; unknown names and non-boolean answers are ignored. The
    aggregate confidence is the **minimum** across selected candidates, so one
    uncertain inclusion rejects the whole shortlist instead of being averaged
    away by confident exclusions. Candidate order is preserved.
    """
    limit = _validate_threshold_value(threshold)
    selected: list[OrchidDecisionAnswer] = []
    for name in candidates:
        answer = result.answers.get(name)
        if answer is None or answer.kind != "boolean" or answer.refusal is not None:
            continue
        if answer.value is True:
            selected.append(answer)

    if not selected:
        return None
    if min(answer.confidence for answer in selected) < limit:
        return None
    return [answer.name for answer in selected]
