"""Decision-backed classification for the ``events`` point.

The ``events`` point asks one advisory question about a *bounded* envelope:
should the deterministic processor pipeline ``process`` the signal, ``skip`` it,
or treat it as a ``dedup`` of something already delivered? An optional
classification label may be requested from the same provider.

Activation is opt-in twice: ``events.enabled`` turns the layer on and
``events.decision.enabled`` (default ``False``) binds the point. With no
binding the processor runs exactly as before and no provider call happens;
every provider failure, refusal, invalid answer or low-confidence answer also
resolves to ``process`` so the deterministic path never loses an event. The
processor evaluates the gate only on the first delivery — a redelivery must run
the existing path, otherwise a provider that flips ``process`` to ``skip``
between attempts could suppress an event that already failed once.

The envelope carries the signal id, type, source, a bounded one-line summary
and a small allowlist of framework-owned metadata. Payload values are **never**
serialised: event payloads routinely carry tickets, documents and credentials,
and a decision provider is an external boundary. The default metadata includes
tenant/user/correlation identifiers and timestamps — never payload values — so
operators must trust the configured provider with those identifiers. The only
other exception is an explicit ``summary`` supplied by a caller that has
already bounded and vetted it.

Exactly one structured ``orchid.decision`` line is emitted per provider call;
a disabled/unconfigured point emits none.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..core.events.signal import Signal
from ..decision.observability import log_decision, safe_log_token

__all__ = [
    "DEFAULT_EVENT_METADATA_KEYS",
    "EVENT_ACTIONS",
    "MAX_CLASSIFICATION_LABELS",
    "MAX_CLASS_LABEL_CHARS",
    "MAX_ENVELOPE_NAME_CHARS",
    "MAX_METADATA_VALUE_CHARS",
    "MAX_SUMMARY_CHARS",
    "EventAction",
    "EventDecision",
    "EventDecisionEnvelope",
    "build_event_envelope",
    "decide_event_action",
]

logger = logging.getLogger(__name__)

EventAction = Literal["process", "skip", "dedup"]

#: Options offered to the provider. ``process`` is first so a provider that
#: defaults to the first option never changes behaviour by accident.
EVENT_ACTIONS: tuple[EventAction, ...] = ("process", "skip", "dedup")

_ACTION_QUESTION = "action"
_CLASSIFICATION_QUESTION = "classification"

#: Bounds for the envelope. Type/source are producer-supplied; the summary is
#: a one-line rendering. Together they keep the JSON state small even when a
#: producer sends a hostile name.
MAX_ENVELOPE_NAME_CHARS = 200
MAX_SUMMARY_CHARS = 280
MAX_METADATA_VALUE_CHARS = 120
MAX_CLASS_LABEL_CHARS = 64
#: Ceiling for the optional ``options.classes`` list so a misconfigured
#: consumer cannot grow the provider prompt without bound.
MAX_CLASSIFICATION_LABELS = 16

#: Framework-owned metadata the envelope may carry. Values are copied through
#: a bounded extractor; an unknown key is ignored, never read from the payload.
DEFAULT_EVENT_METADATA_KEYS: tuple[str, ...] = (
    "tenant_key",
    "user_id",
    "correlation_id",
    "occurred_at",
    "dedupe_key_present",
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")

_ACTION_INSTRUCTIONS = (
    "Choose what the deterministic event pipeline should do with this event. "
    "Choose process when it should flow to its triggers; skip when it should be "
    "acknowledged without running anything; dedup when it repeats an event that "
    "was already delivered. When unsure, choose process."
)


@dataclass(frozen=True)
class EventDecisionEnvelope:
    """The bounded, payload-free view of one signal sent to the provider."""

    signal_id: str
    type: str
    source: str
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_state(self) -> OrchidDecisionState:
        """Render the envelope as the decision state.

        The summary is the state ``text``; identifiers and selected metadata
        live in the JSON-safe payload. No event payload value is included.
        """
        return OrchidDecisionState(
            text=self.summary,
            payload={
                "signal_id": self.signal_id,
                "type": self.type,
                "source": self.source,
                "metadata": dict(self.metadata),
            },
        )


@dataclass(frozen=True)
class EventDecision:
    """The mapped outcome of one ``events`` decision.

    ``source`` distinguishes a confident provider answer (``decision``) from an
    operational fallback (``fallback``) and an unconfigured/disabled point
    (``disabled``). The processor runs the deterministic path for every source
    other than ``decision``; ``fallback_reason`` is a stable code, never
    provider text.
    """

    action: EventAction
    source: Literal["decision", "fallback", "disabled"]
    confidence: float | None = None
    fallback_reason: str | None = None
    provider: str = ""
    classification: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether the provider produced a confident, applicable action."""
        return self.source == "decision"

    @property
    def should_process(self) -> bool:
        """Whether the deterministic pipeline should run normally."""
        return self.action == "process"

    @property
    def gated(self) -> bool:
        """Whether the message should be acknowledged without running triggers."""
        return self.action in ("skip", "dedup")


def build_event_envelope(
    signal: Signal,
    *,
    summary: str | None = None,
    metadata_keys: Sequence[str] | None = None,
) -> EventDecisionEnvelope:
    """Build the bounded envelope for ``signal``.

    ``summary`` defaults to a one-line rendering of the event type and source;
    callers may pass their own bounded summary, but must not pass a raw payload
    value. ``metadata_keys`` is intersected with the framework allowlist — a key
    such as ``"payload"`` is ignored rather than read.
    """
    event_type = _one_line(signal.type, MAX_ENVELOPE_NAME_CHARS) or "unknown"
    source = _one_line(signal.source, MAX_ENVELOPE_NAME_CHARS) or "unknown"
    rendered_summary = (
        _one_line(summary, MAX_SUMMARY_CHARS)
        if summary is not None
        else _one_line(f"event {event_type} from {source}", MAX_SUMMARY_CHARS)
    )

    keys = tuple(metadata_keys) if metadata_keys is not None else DEFAULT_EVENT_METADATA_KEYS
    metadata: dict[str, Any] = {}
    for key in keys:
        extractor = _METADATA_EXTRACTORS.get(key)
        if extractor is None:
            continue
        value = extractor(signal)
        if value is None:
            continue
        metadata[key] = value if isinstance(value, bool) else _one_line(str(value), MAX_METADATA_VALUE_CHARS)

    return EventDecisionEnvelope(
        signal_id=str(signal.signal_id),
        type=event_type,
        source=source,
        summary=rendered_summary,
        metadata=metadata,
    )


async def decide_event_action(
    binding: OrchidDecisionPointBinding | None,
    signal: Signal,
    *,
    summary: str | None = None,
    metadata_keys: Sequence[str] | None = None,
) -> EventDecision:
    """Ask the bound ``events`` point what to do with ``signal``.

    ``binding=None`` means the point is disabled, unconfigured, or its provider
    was not built: no provider call, no decision line, and the deterministic
    processors run (``source="disabled"``).

    Every failure mode — provider error, timeout, refusal, missing/invalid
    answer, low confidence, or an action outside the declared options —
    resolves to ``process`` with a ``fallback`` source and exactly one
    structured fallback line. The helper never raises for an operational
    failure, never retries, and never includes provider text in the result.

    An optional ``classification`` question is asked only when the point's
    ``options.classes`` declares at least two distinct labels. Classification is
    advisory: a missing or low-confidence label never affects the action.
    """
    if binding is None:
        return EventDecision(action="process", source="disabled")

    # Read the threshold once so the action and the advisory classification are
    # judged against the same value even if the binding is dynamic.
    threshold = binding.confidence_threshold
    envelope = build_event_envelope(signal, summary=summary, metadata_keys=metadata_keys)
    labels = _classification_labels(binding)
    questions = _questions(labels)

    started = time.perf_counter()
    try:
        result = await binding.decide(envelope.to_state(), questions)
    except OrchidDecisionError as exc:
        logger.warning("[Events] event decision provider failed code=%s", exc.code)
        return _fallback(
            exc.code or "provider_error",
            binding=binding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=exc.provider,
        )
    except TimeoutError:
        return _fallback(
            "timeout",
            binding=binding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        # Only the exception type crosses this seam: a provider message can
        # contain user state, and a traceback would duplicate it in logs.
        logger.warning("[Events] event decision provider raised %s", type(exc).__name__)
        return _fallback(
            "provider_error",
            binding=binding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        result_provider = result.provider
        answers = result.answers
    except Exception:
        # A wrong return type is a binding contract violation, not a provider
        # failure, but it must still be observable and must not drop the event.
        logger.warning("[Events] event decision provider returned a malformed result")
        return _fallback("invalid_response", binding=binding, latency_ms=latency_ms)

    provider = result_provider if isinstance(result_provider, str) and result_provider else binding.provider
    if not isinstance(answers, Mapping):
        logger.warning("[Events] event decision provider returned a malformed result")
        return _fallback("invalid_response", binding=binding, latency_ms=latency_ms, provider=provider)
    answer = answers.get(_ACTION_QUESTION)
    if not isinstance(answer, OrchidDecisionAnswer):
        return _fallback("invalid_response", binding=binding, latency_ms=latency_ms, provider=provider)
    if answer.refusal is not None:
        return _fallback(
            "refusal",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.kind != "choice":
        return _fallback(
            "invalid_response",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.confidence < threshold:
        return _fallback(
            "low_confidence",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    selected = str(answer.value)
    if selected not in EVENT_ACTIONS:
        return _fallback(
            "unknown_action",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    action = cast(EventAction, selected)
    classification = _classification_answer(
        answers.get(_CLASSIFICATION_QUESTION),
        labels,
        threshold=threshold,
    )
    log_decision(
        point=binding.point,
        path="decision",
        provider=provider,
        latency_ms=latency_ms,
        confidence=answer.confidence,
        outcome="accepted",
    )
    logger.info(
        "[Events] decision selected action=%s classification=%s",
        action,
        safe_log_token(classification) if classification is not None else "none",
    )
    return EventDecision(
        action=action,
        source="decision",
        confidence=answer.confidence,
        provider=provider,
        classification=classification,
    )


def _questions(labels: tuple[str, ...]) -> list[OrchidDecisionQuestion]:
    questions = [
        OrchidDecisionQuestion(
            name=_ACTION_QUESTION,
            kind="choice",
            options=EVENT_ACTIONS,
            instructions=_ACTION_INSTRUCTIONS,
        )
    ]
    if labels:
        questions.append(
            OrchidDecisionQuestion(
                name=_CLASSIFICATION_QUESTION,
                kind="choice",
                options=labels,
                instructions="Optionally label the event with one of the configured classes.",
                required=False,
            )
        )
    return questions


def _classification_labels(
    binding: OrchidDecisionPointBinding,
) -> tuple[str, ...]:
    """Return the configured classification labels, or ``()`` when unset/invalid.

    ``options.classes`` is operator-supplied YAML. A malformed declaration is
    ignored rather than failing the event: classification is advisory and the
    deterministic action must still be available.
    """
    raw = binding.options.get("classes")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        return ()
    labels: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        label = _one_line(item, MAX_CLASS_LABEL_CHARS)
        if not label or label in labels:
            continue
        labels.append(label)
        if len(labels) >= MAX_CLASSIFICATION_LABELS:
            break
    return tuple(labels) if len(labels) >= 2 else ()


def _classification_answer(
    answer: Any,
    labels: tuple[str, ...],
    *,
    threshold: float,
) -> str | None:
    """Return a confident declared label, or ``None``.

    A missing, refused, low-confidence or unknown label is dropped without
    changing the accepted action.
    """
    if (
        not labels
        or not isinstance(answer, OrchidDecisionAnswer)
        or answer.refusal is not None
        or answer.kind != "choice"
    ):
        return None
    if answer.confidence < threshold:
        return None
    selected = str(answer.value)
    return selected if selected in labels else None


def _fallback(
    reason: str,
    *,
    binding: OrchidDecisionPointBinding,
    latency_ms: float | None = None,
    confidence: float | None = None,
    provider: str = "",
) -> EventDecision:
    """Emit one structured fallback line and run the deterministic processors."""
    resolved_provider = provider or binding.provider
    logger.warning("[Events] event decision fell back to the deterministic processors (reason=%s)", reason)
    log_decision(
        point=binding.point,
        path="llm_fallback",
        provider=resolved_provider,
        latency_ms=latency_ms,
        confidence=confidence,
        outcome="fallback",
        fallback_reason=reason,
    )
    return EventDecision(
        action="process",
        source="fallback",
        confidence=confidence,
        fallback_reason=reason,
        provider=resolved_provider,
    )


def _one_line(value: str, limit: int) -> str:
    """Reduce ``value`` to one bounded line without raising on hostile input."""
    text = value if isinstance(value, str) else str(value)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = " ".join(text.split())
    return text[:limit]


_METADATA_EXTRACTORS: dict[str, Any] = {
    "tenant_key": lambda signal: signal.tenant_key,
    "user_id": lambda signal: signal.user_id,
    "correlation_id": lambda signal: signal.correlation_id,
    "occurred_at": lambda signal: signal.occurred_at.isoformat(),
    "dedupe_key_present": lambda signal: signal.dedupe_key is not None,
    "relay_status": lambda signal: signal.relay_status,
}
