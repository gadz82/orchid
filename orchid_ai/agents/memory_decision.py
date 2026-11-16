"""Decision-backed conversation-memory actions for the ``memory`` point.

The helper answers one advisory question about a *bounded* recent-turn state:
should the framework ``summarize`` the older turns, ``keep`` the current state,
or ``persist`` a memory entry? It never writes storage itself — the caller maps
the action onto the existing summary/storage APIs — and it never changes the
configured heuristic when no confident decision arrives.

Every outcome other than a confident, available action resolves to ``keep`` so
the caller's configured memory strategy stays in control. The request state
carries counts, roles and truncated user/assistant excerpts: framework secrets,
system messages and tool output never cross this seam. User/assistant content
is sent bounded but **without redaction** — summarisation needs it, and the
operator is responsible for trusting the configured provider with that content.
Exactly one structured ``orchid.decision`` line is emitted per provider call;
an unconfigured/disabled point emits none.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..decision.observability import log_decision

__all__ = [
    "MAX_PAYLOAD_CHARS",
    "MAX_STATE_TURNS",
    "MAX_SUMMARY_CHARS",
    "MAX_TURN_CHARS",
    "MEMORY_ACTIONS",
    "MemoryAction",
    "MemoryDecision",
    "build_memory_decision_state",
    "decide_memory_action",
]

logger = logging.getLogger(__name__)

MemoryAction = Literal["summarize", "keep", "persist"]

#: Options offered to the provider. ``keep`` is first so a provider that
#: defaults to the first option never changes behaviour by accident.
MEMORY_ACTIONS: tuple[MemoryAction, ...] = ("keep", "persist", "summarize")

_POINT = "memory"
_QUESTION = "action"

#: Bounds for the decision state. Together they keep the JSON payload well
#: below the transport cap even when every excerpt is fully truncated.
MAX_STATE_TURNS = 6
MAX_TURN_CHARS = 240
MAX_SUMMARY_CHARS = 600
MAX_PAYLOAD_CHARS = 2048

_INSTRUCTIONS = (
    "Choose what to do with conversation memory for the next turn. "
    "Choose keep when no change is needed; summarize to compress older turns "
    "through the existing summary path; persist to write a memory entry "
    "through the existing storage API. Never choose an action marked unavailable."
)


@dataclass(frozen=True)
class MemoryDecision:
    """The mapped outcome of one ``memory`` decision.

    ``source`` distinguishes a confident provider answer (``decision``) from an
    operational fallback (``fallback``) and an unconfigured/disabled point
    (``disabled``). Callers run their configured heuristic for every source
    other than ``decision``. ``fallback_reason`` is a stable code, never
    provider text.
    """

    action: MemoryAction
    source: Literal["decision", "fallback", "disabled"]
    confidence: float | None = None
    fallback_reason: str | None = None
    provider: str = ""

    @property
    def accepted(self) -> bool:
        """Whether the provider produced a confident, applicable action."""
        return self.source == "decision"


def build_memory_decision_state(
    recent_turns: Sequence[Mapping[str, Any]],
    running_summary: str | None = None,
    *,
    persist_available: bool = False,
    summarize_available: bool = True,
) -> OrchidDecisionState:
    """Build the bounded JSON-safe state sent to the memory decision provider.

    Only ``user``/``assistant`` turns are eligible; tool, system and unknown
    roles are dropped so raw tool output cannot reach an external provider.
    Bounded user/assistant excerpts **are** included deliberately — a
    summarisation decision needs them — and no redaction is applied, so the
    operator must trust the configured provider with that content. Content is
    truncated per turn, the summary is truncated separately, the turn count is
    capped, and the total payload is shrunk from the oldest turn until it fits
    :data:`MAX_PAYLOAD_CHARS`.

    The free-text ``text`` field stays empty: the decision is about the shape
    and size of the memory state, not the request itself.
    """
    selected = [
        turn for turn in recent_turns if isinstance(turn, Mapping) and turn.get("role") in ("user", "assistant")
    ]
    eligible_turns = len(selected)
    selected = selected[-MAX_STATE_TURNS:]

    summary = running_summary if isinstance(running_summary, str) else ""
    base: dict[str, Any] = {
        "turn_count": 0,
        "has_more_turns": False,
        "persist_available": bool(persist_available),
        "summarize_available": bool(summarize_available),
        "summary_present": bool(summary),
        "summary_chars": len(summary),
        "summary_excerpt": summary[:MAX_SUMMARY_CHARS] or None,
    }

    entries: list[dict[str, Any]] = []
    for turn in reversed(selected):
        content = turn.get("content")
        text = content if isinstance(content, str) else ""
        entry: dict[str, Any] = {
            "role": turn.get("role"),
            "chars": len(text),
            "truncated": len(text) > MAX_TURN_CHARS,
            "excerpt": text[:MAX_TURN_CHARS],
        }
        if len(json.dumps({**base, "turns": [entry, *entries]})) > MAX_PAYLOAD_CHARS:
            break
        entries.append(entry)

    entries.reverse()
    base["turn_count"] = len(entries)
    base["has_more_turns"] = eligible_turns > len(entries)
    return OrchidDecisionState(text="", payload={**base, "turns": entries})


async def decide_memory_action(
    binding: OrchidDecisionPointBinding | None,
    *,
    recent_turns: Sequence[Mapping[str, Any]],
    running_summary: str | None = None,
    persist_available: bool = False,
    summarize_available: bool = True,
) -> MemoryDecision:
    """Ask the bound ``memory`` point which action to take.

    ``binding=None`` means the point is disabled, unconfigured, or its provider
    was not built: no provider call, no decision line, and the caller's
    configured heuristic remains in control (``source="disabled"``).

    Every failure mode — provider error, timeout, refusal, missing/invalid
    answer, low confidence, or an action that is not available in this context
    — resolves to ``keep`` with a ``fallback`` source and exactly one
    structured fallback line. The helper never raises for an operational
    failure, never retries, and never includes provider text in the result.

    A fallback means "no additional decision-driven action": the caller then
    runs its configured heuristic, which may still perform its normal legacy
    write. An accepted ``persist`` is logged before the caller writes, and the
    caller swallows storage failures, so the accepted line does not prove a
    successful write; a future change should log after the write or add a
    failure signal.
    """
    if binding is None:
        return MemoryDecision(action="keep", source="disabled")

    state = build_memory_decision_state(
        recent_turns,
        running_summary,
        persist_available=persist_available,
        summarize_available=summarize_available,
    )
    question = OrchidDecisionQuestion(
        name=_QUESTION,
        kind="choice",
        options=MEMORY_ACTIONS,
        instructions=_INSTRUCTIONS,
    )

    started = time.perf_counter()
    try:
        result = await binding.decide(state, [question])
    except OrchidDecisionError as exc:
        logger.warning("[Memory] memory decision provider failed code=%s", exc.code)
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
        logger.warning("[Memory] memory decision provider raised %s", type(exc).__name__)
        return _fallback(
            "provider_error",
            binding=binding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    provider = result.provider or binding.provider
    answer = result.answers.get(_QUESTION)
    if answer is None:
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
    if answer.confidence < binding.confidence_threshold:
        return _fallback(
            "low_confidence",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    selected = str(answer.value)
    if selected not in MEMORY_ACTIONS:
        return _fallback(
            "unknown_action",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    action = cast(MemoryAction, selected)
    if action == "persist" and not persist_available:
        return _fallback(
            "persist_unavailable",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if action == "summarize" and not summarize_available:
        return _fallback(
            "summarize_unavailable",
            binding=binding,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    log_decision(
        point=_POINT,
        path="decision",
        provider=provider,
        latency_ms=latency_ms,
        confidence=answer.confidence,
        outcome="accepted",
    )
    logger.info("[Memory] decision selected action=%s", action)
    return MemoryDecision(
        action=action,
        source="decision",
        confidence=answer.confidence,
        provider=provider,
    )


def _fallback(
    reason: str,
    *,
    binding: OrchidDecisionPointBinding,
    latency_ms: float | None = None,
    confidence: float | None = None,
    provider: str = "",
) -> MemoryDecision:
    """Emit one structured fallback line and keep the configured heuristic.

    The caller may still run its normal legacy summary/persistence path; the
    fallback only guarantees that the decision adds no action of its own.
    """
    resolved_provider = provider or binding.provider
    logger.warning("[Memory] memory decision fell back to the configured memory (reason=%s)", reason)
    log_decision(
        point=_POINT,
        path="llm_fallback",
        provider=resolved_provider,
        latency_ms=latency_ms,
        confidence=confidence,
        outcome="fallback",
        fallback_reason=reason,
    )
    return MemoryDecision(
        action="keep",
        source="fallback",
        confidence=confidence,
        fallback_reason=reason,
        provider=resolved_provider,
    )
