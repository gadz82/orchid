"""Shared answer validation and construction for decision adapters.

Both wire dialects ultimately answer the same three question kinds. Keeping the
validation here means a choice answer without confidence evidence fails the same
way whether it arrived as an object map or an array, and the contract suite can
compare the two adapters on behaviour rather than syntax.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from ..core.decision import OrchidDecisionAnswer, OrchidDecisionError, OrchidDecisionQuestion

__all__ = [
    "ErrorFactory",
    "check_answer_type",
    "finite_number",
    "normalise_boolean",
    "normalise_choice",
    "normalise_score",
    "normalise_usage",
    "refusal_answer",
    "sanitise_reason",
    "validate_probabilities",
]

ErrorFactory = Callable[[str], OrchidDecisionError]

_MAX_REFUSAL_CHARS = 500


def finite_number(value: Any) -> float | None:
    """Return a finite float, or ``None`` for non-finite/non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check_answer_type(
    question: OrchidDecisionQuestion,
    answer_type: Any,
    expected: str,
    error: ErrorFactory,
) -> None:
    """Reject an answer whose wire type contradicts the question kind."""
    if answer_type is None:
        return
    if not isinstance(answer_type, str):
        raise error("answer type must be a string")
    if answer_type not in (expected, "refusal"):
        raise error("answer type does not match the question kind")


def normalise_boolean(
    question: OrchidDecisionQuestion,
    probability: Any,
    error: ErrorFactory,
) -> OrchidDecisionAnswer:
    number = finite_number(probability)
    if number is None or not 0.0 <= number <= 1.0:
        raise error("boolean probability must be finite and within [0, 1]")
    value = number >= 0.5
    return OrchidDecisionAnswer(
        name=question.name,
        kind="boolean",
        value=value,
        confidence=number if value else 1.0 - number,
        probabilities={"true": number, "false": 1.0 - number},
    )


def normalise_choice(
    question: OrchidDecisionQuestion,
    selected: Any,
    confidence: Any,
    probabilities: Any,
    error: ErrorFactory,
) -> OrchidDecisionAnswer:
    if not isinstance(selected, str) or selected not in question.options:
        raise error("choice answer is not a declared option")

    explicit_confidence = _optional_confidence(confidence, error)
    distribution = validate_probabilities(probabilities, expected=set(question.options), error=error)
    if explicit_confidence is None:
        if selected not in distribution:
            raise error("choice answer is missing confidence evidence")
        explicit_confidence = distribution[selected]

    return OrchidDecisionAnswer(
        name=question.name,
        kind="choice",
        value=selected,
        confidence=explicit_confidence,
        probabilities=distribution,
    )


def normalise_score(
    question: OrchidDecisionQuestion,
    score: Any,
    confidence: Any,
    probabilities: Any,
    error: ErrorFactory,
) -> OrchidDecisionAnswer:
    number = finite_number(score)
    if number is None:
        raise error("score answer must be a finite number")
    upper = float(len(question.rubric) - 1)
    if not 0.0 <= number <= upper:
        raise error("score answer is outside the rubric range")

    explicit_confidence = _optional_confidence(confidence, error)
    expected = set(question.rubric) | {str(index) for index in range(len(question.rubric))}
    distribution = validate_probabilities(probabilities, expected=expected, error=error)
    if explicit_confidence is None:
        if not distribution:
            raise error("score answer is missing confidence evidence")
        explicit_confidence = max(distribution.values())

    return OrchidDecisionAnswer(
        name=question.name,
        kind="score",
        value=number,
        confidence=explicit_confidence,
        probabilities=distribution,
    )


def refusal_answer(question: OrchidDecisionQuestion, reason: Any) -> OrchidDecisionAnswer:
    text = reason if isinstance(reason, str) and reason.strip() else "refused"
    return OrchidDecisionAnswer(
        name=question.name,
        kind=question.kind,
        value=None,
        confidence=0.0,
        refusal=sanitise_reason(text),
    )


def validate_probabilities(probabilities: Any, *, expected: set[str], error: ErrorFactory) -> dict[str, float]:
    if probabilities is None:
        return {}
    if not isinstance(probabilities, Mapping):
        raise error("probabilities must be an object")
    result: dict[str, float] = {}
    for key, value in probabilities.items():
        if not isinstance(key, str) or key not in expected:
            continue
        number = finite_number(value)
        if number is None or not 0.0 <= number <= 1.0:
            raise error(f"probability {key!r} must be finite and within [0, 1]")
        result[key] = number
    return result


def normalise_usage(value: Any, *, keys: frozenset[str]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, int] = {}
    for key, raw in value.items():
        if key not in keys:
            continue
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            usage[key] = raw
    return usage


def sanitise_reason(value: str) -> str:
    printable = "".join(character for character in value if character.isprintable() or character in "\n\t")
    return printable.strip()[:_MAX_REFUSAL_CHARS]


def _optional_confidence(value: Any, error: ErrorFactory) -> float | None:
    if value is None:
        return None
    number = finite_number(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise error("confidence must be finite and within [0, 1]")
    return number
