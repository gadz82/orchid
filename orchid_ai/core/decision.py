"""Decision-model abstractions — typed judgements instead of generated text.

A decision model evaluates a *state* against named *questions* and returns
typed answers with calibrated confidence (boolean, choice, or score). It does
not generate prose. Providers speak different wire dialects, but every
consumer in the framework depends on the single contract defined here.

This module is deliberately stdlib-only: transport, provider dialects, and
configuration live outside ``core/``.
"""

from __future__ import annotations

import copy
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

__all__ = [
    "DecisionErrorCode",
    "DecisionKind",
    "OrchidDecisionAnswer",
    "OrchidDecisionError",
    "OrchidDecisionFallbackPolicy",
    "OrchidDecisionModel",
    "OrchidDecisionPointBinding",
    "OrchidDecisionPointScope",
    "OrchidDecisionPointSpec",
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
OrchidDecisionPointScope = Literal["fleet", "agent"]

DecisionErrorCode = Literal[
    "not_configured",
    "timeout",
    "provider_unavailable",
    "egress_denied",
    "invalid_response",
    "refusal",
    "budget_exhausted",
]

_DECISION_KINDS: frozenset[str] = frozenset(get_args(DecisionKind))
_DECISION_ERROR_CODES: frozenset[str] = frozenset(get_args(DecisionErrorCode))
_DECISION_POINT_SCOPES: frozenset[str] = frozenset(get_args(OrchidDecisionPointScope))

_POINT_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_FALLBACK_MODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

_MAX_PAYLOAD_DEPTH = 20


def _as_finite_float(value: Any, *, field_name: str) -> float:
    """Return ``value`` as a finite float, or raise a typed error.

    ``float(10**400)`` raises ``OverflowError``, which is neither ``TypeError``
    nor ``ValueError``. A provider can send arbitrarily large integers, so the
    conversion is centralised here and normalised into the documented error
    contract callers use for fallback.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} is too large to represent as a float") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _bounded_confidence(value: Any, *, field_name: str) -> float:
    number = _as_finite_float(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within [0, 1], got {value!r}")
    return number


def _validate_probabilities(probabilities: Any) -> dict[str, float]:
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")
    copied: dict[str, float] = {}
    for key, value in probabilities.items():
        if not isinstance(key, str):
            raise TypeError("probability keys must be strings")
        copied[key] = _bounded_confidence(value, field_name=f"probability[{key!r}]")
    return copied


def _validate_kind(kind: Any) -> None:
    if not isinstance(kind, str):
        raise TypeError(f"kind must be a string, got {type(kind).__name__}")
    if kind not in _DECISION_KINDS:
        raise ValueError(f"kind must be one of {sorted(_DECISION_KINDS)}, got {kind!r}")


def _validate_json_value(value: Any, *, path: str = "$", depth: int = 0) -> None:
    """Reject payload values a JSON transport cannot represent.

    The state crosses a JSON boundary, so a non-string key, a set, bytes, or a
    non-finite float must fail here rather than after the request starts.
    """
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValueError(f"payload exceeds maximum nesting depth at {path}")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"payload float at {path} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"payload key at {path} must be a string")
            _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        raise TypeError(f"payload mapping at {path} must be a plain dict")
    raise TypeError(f"payload value at {path} is not JSON-safe: {type(value).__name__}")


def _validate_label_sequence(values: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return tuple(values)


@dataclass(frozen=True)
class OrchidDecisionState:
    """The context a decision model evaluates.

    ``payload`` carries structured context (candidates, passages, counts). It is
    validated as JSON-safe and deep-copied on construction so a caller cannot
    mutate the state while an async provider call is in flight.
    """

    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("state text must be a string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("state payload must be a mapping")
        payload = dict(self.payload)
        _validate_json_value(payload)
        object.__setattr__(self, "payload", copy.deepcopy(payload))


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
        if not isinstance(self.name, str):
            raise TypeError("question name must be a string")
        if not self.name.strip():
            raise ValueError("question name must be non-empty")
        if not isinstance(self.instructions, str):
            raise TypeError("question instructions must be a string")
        if not isinstance(self.required, bool):
            raise TypeError("question required must be a boolean")

        options = _validate_label_sequence(self.options, field_name="options")
        rubric = _validate_label_sequence(self.rubric, field_name="rubric")
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
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} entries must be strings")
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


@dataclass(frozen=True)
class OrchidDecisionAnswer:
    """The normalised result for one question.

    A refusal is a valid answer: ``value`` is ``None``, ``refusal`` explains why,
    and sibling answers remain usable. ``confidence`` is always finite and within
    ``[0, 1]`` so a caller can compare it with a threshold without defensive
    checks.
    """

    name: str
    kind: DecisionKind
    value: bool | str | float | None
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    refusal: str | None = None

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if not isinstance(self.name, str):
            raise TypeError("answer name must be a string")
        if not self.name.strip():
            raise ValueError("answer name must be non-empty")

        object.__setattr__(
            self,
            "confidence",
            _bounded_confidence(self.confidence, field_name="confidence"),
        )
        object.__setattr__(self, "probabilities", _validate_probabilities(self.probabilities))

        if self.refusal is not None:
            if not isinstance(self.refusal, str):
                raise TypeError("refusal must be a string")
            if self.value is not None:
                raise ValueError("a refused answer must not carry a value")
            return

        if self.kind == "boolean":
            if not isinstance(self.value, bool):
                raise TypeError("boolean answers require a bool value")
            return

        if self.kind == "choice":
            if not isinstance(self.value, str):
                raise TypeError("choice answers require a string value")
            if not self.value.strip():
                raise ValueError("choice answers require a non-empty value")
            return

        number = _as_finite_float(self.value, field_name="score answer")
        object.__setattr__(self, "value", number)


@dataclass(frozen=True)
class OrchidDecisionResult:
    """Normalised answers for one decision request."""

    answers: Mapping[str, OrchidDecisionAnswer]
    provider: str
    model: str
    latency_ms: float = 0.0
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str):
            raise TypeError("result provider must be a string")
        if not self.provider.strip():
            raise ValueError("result provider must be non-empty")
        if not isinstance(self.model, str):
            raise TypeError("result model must be a string")
        if not self.model.strip():
            raise ValueError("result model must be non-empty")

        latency = _as_finite_float(self.latency_ms, field_name="latency_ms")
        if latency < 0.0:
            raise ValueError("latency_ms must be non-negative")
        object.__setattr__(self, "latency_ms", latency)

        if not isinstance(self.answers, Mapping):
            raise TypeError("answers must be a mapping")
        copied: dict[str, OrchidDecisionAnswer] = {}
        for key, answer in self.answers.items():
            if not isinstance(key, str) or not isinstance(answer, OrchidDecisionAnswer):
                raise TypeError("answers must map string names to OrchidDecisionAnswer values")
            if key != answer.name:
                raise ValueError(f"answer key {key!r} does not match answer name {answer.name!r}")
            copied[key] = answer
        object.__setattr__(self, "answers", copied)

        if not isinstance(self.usage, Mapping):
            raise TypeError("usage must be a mapping")
        usage: dict[str, int] = {}
        for key, value in self.usage.items():
            if not isinstance(key, str):
                raise TypeError("usage keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"usage[{key!r}] must be an integer")
            if value < 0:
                raise ValueError(f"usage[{key!r}] must be non-negative")
            usage[key] = value
        object.__setattr__(self, "usage", usage)


class OrchidDecisionModel(ABC):
    """Provider-agnostic contract for decision models.

    # ORCHID-MEM: 01a1a102-9e83-7b0a-8a8e-333d3a323039 — Provider-agnostic typed decision contract.

    Implementations translate :class:`OrchidDecisionQuestion` values into their
    provider's wire shape and normalise the response. They must not perform side
    effects, compose tool arguments, or decide workflow on their own: they answer
    questions, and the framework owns the branch.

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

    async def aclose(self) -> None:
        """Release transport resources held by this provider.

        The default is a no-op so an in-process provider needs no cleanup. The
        composition root calls this uniformly on reload and teardown.
        """
        return


@dataclass(frozen=True)
class OrchidDecisionFallbackPolicy:
    """What a consumer does when a point yields no usable answer.

    ``mode`` is a stable, domain-neutral code describing the fallback target
    (``llm``, ``default``, ``all_tools``, ``configured_strategy``, …). Consumers
    own the implementation; the spec only declares it so operators can see the
    contract and tests can assert it.
    """

    mode: str = "default"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("fallback mode must be a string")
        if not _FALLBACK_MODE_RE.fullmatch(self.mode):
            raise ValueError(f"fallback mode {self.mode!r} must be a lowercase identifier")
        if not isinstance(self.description, str):
            raise TypeError("fallback description must be a string")


PointConfigValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class OrchidDecisionPointSpec:
    """Declarative description of one decision point.

    A point spec is *data*: it names the point, says whether the binding is
    fleet-wide or per agent, records the hard candidate ceiling and the declared
    fallback, and optionally carries a validator for point-specific YAML. It
    contains no flow logic — adapters own result mapping and fallbacks.

    Validation of the optional ``validate_config`` callable is deliberately
    delegated: it receives the raw, explicitly-set point mapping and must raise
    ``ValueError`` on invalid input.
    """

    name: str
    scope: OrchidDecisionPointScope = "agent"
    max_candidates: int | None = None
    fallback: OrchidDecisionFallbackPolicy | str = field(default_factory=OrchidDecisionFallbackPolicy)
    description: str = ""
    builtin: bool = False
    validate_config: PointConfigValidator | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("decision point name must be a string")
        if not _POINT_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"decision point name {self.name!r} must be a lowercase identifier "
                "(letters, digits, underscore; at most 64 characters)"
            )
        if not isinstance(self.scope, str):
            raise TypeError("decision point scope must be a string")
        if self.scope not in _DECISION_POINT_SCOPES:
            raise ValueError(
                f"decision point scope must be one of {sorted(_DECISION_POINT_SCOPES)}, got {self.scope!r}"
            )
        if self.max_candidates is not None:
            if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int):
                raise TypeError("decision point max_candidates must be an integer or None")
            if self.max_candidates < 1:
                raise ValueError("decision point max_candidates must be at least 1")

        fallback = self.fallback
        if isinstance(fallback, str):
            fallback = OrchidDecisionFallbackPolicy(mode=fallback)
        if not isinstance(fallback, OrchidDecisionFallbackPolicy):
            raise TypeError("decision point fallback must be an OrchidDecisionFallbackPolicy or a mode string")
        object.__setattr__(self, "fallback", fallback)

        if not isinstance(self.description, str):
            raise TypeError("decision point description must be a string")
        if not isinstance(self.builtin, bool):
            raise TypeError("decision point builtin must be a boolean")
        if self.validate_config is not None and not callable(self.validate_config):
            raise TypeError("decision point validate_config must be callable or None")


@runtime_checkable
class OrchidDecisionPointBinding(Protocol):
    """Narrow interface a consumer receives for one resolved decision point.

    Consumers depend on this protocol — never on a concrete provider, transport,
    or config model. ``decide`` mirrors :class:`OrchidDecisionModel` so the
    binding is a drop-in replacement for existing call sites, while ``metadata``
    and the limits are read-only declarations for the consumer and the
    observability helper.
    """

    @property
    def point(self) -> str:
        """Registered point name (for example ``"tool_choice"``)."""
        ...

    @property
    def spec(self) -> OrchidDecisionPointSpec:
        """Declarative point specification."""
        ...

    @property
    def provider(self) -> str:
        """Provider key selected in configuration, or ``""`` when unset."""
        ...

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence a consumer should require before accepting."""
        ...

    @property
    def max_candidates(self) -> int | None:
        """Effective candidate ceiling, or ``None`` when unbounded."""
        ...

    @property
    def options(self) -> Mapping[str, Any]:
        """Point-specific parameters resolved from YAML (read-only).

        Consumers may read this mapping but must not mutate it; the concrete
        binding returns an immutable view.
        """
        ...

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Sequence[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        """Answer ``questions`` about ``state`` through the bound provider."""
        ...

    def metadata(self) -> Mapping[str, Any]:
        """Safe, structured fields for graph metadata and decision logs."""
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
        if not isinstance(message, str):
            raise TypeError("error message must be a string")
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
    return _bounded_confidence(threshold, field_name="threshold")


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
    limit = validate_threshold(threshold)
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
    decision. An empty request returns ``None`` for the same reason.
    """
    limit = validate_threshold(threshold)
    selected: dict[str, OrchidDecisionAnswer] = {}
    for name in names:
        answer = require_confidence(result, name, limit)
        if answer is None:
            return None
        selected[name] = answer
    return selected or None


def aggregate_shortlist(
    result: OrchidDecisionResult,
    candidates: Sequence[str],
    *,
    threshold: float = 0.5,
) -> list[str] | None:
    """Return the candidates a boolean decision selected, or ``None``.

    Each candidate is a separate boolean question. Only ``True`` answers are
    considered; unknown names and non-boolean answers are ignored. The aggregate
    confidence is the **minimum** across selected candidates, so one uncertain
    inclusion rejects the whole shortlist instead of being averaged away by
    confident exclusions. Candidate order is preserved and duplicates are
    removed.
    """
    limit = validate_threshold(threshold)
    selected: list[OrchidDecisionAnswer] = []
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
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
