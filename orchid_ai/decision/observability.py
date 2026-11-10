"""Structured observability for decision points.

Every point — legacy or registered — emits *one* line through
:func:`log_decision`. The helper accepts exactly the public observability
contract (point, path, provider, latency, confidence, outcome, fallback reason)
and deliberately has no payload argument: raw user text, document bodies, event
payloads and secrets cannot leak through this seam.

The emitted record is also returned so callers can store the same fields in
graph metadata without re-formatting them.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

__all__ = [
    "DECISION_LOGGER_NAME",
    "DECISION_LOG_FIELDS",
    "build_decision_record",
    "log_decision",
    "safe_log_token",
]

DECISION_LOGGER_NAME = "orchid.decision"

#: Field order of the single structured decision line.
DECISION_LOG_FIELDS = (
    "point",
    "path",
    "provider",
    "latency_ms",
    "confidence",
    "outcome",
    "fallback_reason",
)

_PATHS = frozenset({"decision", "llm_fallback"})
_OUTCOMES = frozenset({"accepted", "fallback"})
_LOG_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_MAX_TOKEN_CHARS = 128

_logger = logging.getLogger(DECISION_LOGGER_NAME)


def build_decision_record(
    *,
    point: str,
    outcome: str,
    path: str = "decision",
    provider: str = "",
    latency_ms: float | None = None,
    confidence: float | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Validate and normalise one decision observation.

    Raises ``ValueError`` for a programmer error (unknown outcome/path, negative
    latency, out-of-range confidence). The returned dict preserves
    :data:`DECISION_LOG_FIELDS` order and contains no free-form content.
    """
    if not isinstance(point, str) or not point.strip():
        raise ValueError("decision log point must be a non-empty string")
    if path not in _PATHS:
        raise ValueError(f"decision log path must be one of {sorted(_PATHS)}, got {path!r}")
    if outcome not in _OUTCOMES:
        raise ValueError(f"decision log outcome must be one of {sorted(_OUTCOMES)}, got {outcome!r}")
    if not isinstance(provider, str):
        raise TypeError("decision log provider must be a string")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise TypeError("decision log fallback_reason must be a string or None")

    return {
        "point": safe_log_token(point),
        "path": path,
        "provider": safe_log_token(provider),
        "latency_ms": _normalise_latency(latency_ms),
        "confidence": _normalise_confidence(confidence),
        "outcome": outcome,
        "fallback_reason": safe_log_token(fallback_reason) if fallback_reason else None,
    }


def log_decision(
    *,
    point: str,
    outcome: str,
    path: str = "decision",
    provider: str = "",
    latency_ms: float | None = None,
    confidence: float | None = None,
    fallback_reason: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Emit one structured decision line and return its fields.

    The line always carries :data:`DECISION_LOG_FIELDS`. Values are reduced to
    a safe single-line token, so a provider key or fallback code cannot inject
    extra log lines.
    """
    record = build_decision_record(
        point=point,
        outcome=outcome,
        path=path,
        provider=provider,
        latency_ms=latency_ms,
        confidence=confidence,
        fallback_reason=fallback_reason,
    )
    (logger or _logger).info(
        "decision point=%s path=%s provider=%s latency_ms=%s confidence=%s outcome=%s fallback_reason=%s",
        record["point"],
        record["path"],
        record["provider"],
        _format_number(record["latency_ms"]),
        _format_number(record["confidence"]),
        record["outcome"],
        record["fallback_reason"] if record["fallback_reason"] is not None else "null",
    )
    return record


def _normalise_latency(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("decision log latency_ms must be a number or None")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("decision log latency_ms must be finite and non-negative")
    return round(number, 3)


def _normalise_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("decision log confidence must be a number or None")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("decision log confidence must be finite and within [0, 1]")
    return round(number, 4)


def safe_log_token(value: str) -> str:
    """Reduce a value to one safe log token; never raise on hostile input.

    Shared by :func:`log_decision` and binding metadata so a provider key or
    fallback code cannot inject whitespace, newlines, or ``=``-separated fields
    into a structured line.
    """
    token = _LOG_TOKEN_RE.sub("_", value.strip())[:_MAX_TOKEN_CHARS]
    return token or "unknown"


def _format_number(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"
