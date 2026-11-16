"""Decision-backed parser and ingestion-strategy selection.

The ``ingestion`` point may choose a *registered* parser before the file is
parsed and a *registered* ingestion strategy after the text has been extracted.
It can never invent a parser or strategy: candidates come from the parser and
strategy registries, optionally narrowed by ``options.parsers`` /
``options.strategies``. Any unknown, refused, low-confidence or failed answer
falls back to the existing extension/config default, so an unconfigured point
is byte-for-byte the legacy path.

The provider sees a bounded :class:`DocumentDescriptor` — a sanitised basename,
extension, MIME type, byte size, and an optional short text snapshot. The full
document body never crosses the decision boundary, and the structured decision
line carries no file content.

There is no separate chunker registry: choosing an ingestion strategy also
chooses its chunker. The parser and strategy phases both run under the
``ingestion`` point, so only the question name and instructions distinguish
them; pre-call fallbacks (invalid options, too many/no candidates) emit a
structured line with ``latency_ms=null``, while a single-entry allowlist is
selected locally without a provider call or decision line. Exactly one
structured line is emitted per provider call; a disabled point emits none.
"""

from __future__ import annotations

import logging
import mimetypes
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..decision.observability import log_decision
from .parsers import list_parser_names
from .strategies import INGESTION_REGISTRY

__all__ = [
    "MAX_CANDIDATES",
    "MAX_EXTENSION_CHARS",
    "MAX_FILENAME_CHARS",
    "MAX_MIME_CHARS",
    "MAX_SNAPSHOT_CHARS",
    "DocumentDescriptor",
    "DocumentSelection",
    "select_document_parser",
    "select_ingestion_strategy",
]

logger = logging.getLogger(__name__)

#: Bounds for the descriptor. A hostile filename or provider key cannot grow
#: the request state without bound.
MAX_FILENAME_CHARS = 200
MAX_EXTENSION_CHARS = 16
MAX_MIME_CHARS = 100
MAX_SNAPSHOT_CHARS = 500
#: Hard candidate ceiling for a choice question. Both registries are small, but
#: an integrator can register many candidates and the provider prompt must stay
#: bounded.
MAX_CANDIDATES = 32

_PARSER_QUESTION = "parser"
_STRATEGY_QUESTION = "strategy"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")

_PARSER_INSTRUCTIONS = (
    "Choose the registered parser that should extract text from this document. "
    "Prefer the parser that matches the document format; choose another parser "
    "only when the file's name or metadata indicates a different format."
)

_STRATEGY_INSTRUCTIONS = (
    "Choose the registered ingestion strategy that should chunk this document. "
    "Prefer the simplest strategy that preserves the document structure; choose "
    "a hierarchical or semantic strategy only when the content needs it."
)


@dataclass(frozen=True)
class DocumentDescriptor:
    """Bounded metadata describing one document to the decision provider.

    ``filename`` is a sanitised basename — never a path. ``text_snapshot`` is an
    optional, whitespace-collapsed first-N excerpt used to judge the strategy
    after extraction; it is empty during parser selection and is never the full
    document body.
    """

    filename: str
    extension: str
    mime_type: str
    size_bytes: int
    text_snapshot: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str):
            raise TypeError("document filename must be a string")
        if not isinstance(self.extension, str):
            raise TypeError("document extension must be a string")
        if not isinstance(self.mime_type, str):
            raise TypeError("document mime_type must be a string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("document size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("document size_bytes must be non-negative")
        if not isinstance(self.text_snapshot, str):
            raise TypeError("document text_snapshot must be a string")
        if len(self.filename) > MAX_FILENAME_CHARS:
            raise ValueError(f"document filename exceeds {MAX_FILENAME_CHARS} characters")
        if len(self.extension) > MAX_EXTENSION_CHARS:
            raise ValueError(f"document extension exceeds {MAX_EXTENSION_CHARS} characters")
        if len(self.mime_type) > MAX_MIME_CHARS:
            raise ValueError(f"document mime_type exceeds {MAX_MIME_CHARS} characters")
        if len(self.text_snapshot) > MAX_SNAPSHOT_CHARS:
            raise ValueError(f"document text_snapshot exceeds {MAX_SNAPSHOT_CHARS} characters")

    @classmethod
    def from_bytes(
        cls,
        *,
        file_bytes: bytes,
        filename: str,
        text: str | None = None,
        snapshot_chars: int = MAX_SNAPSHOT_CHARS,
    ) -> DocumentDescriptor:
        """Build a bounded descriptor from file bytes and an extracted text.

        ``text`` is optional. When supplied, only the first ``snapshot_chars``
        characters (after whitespace collapsing) are kept; the caller's full
        text is never stored or copied by the descriptor.
        """
        if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("file_bytes must be bytes-like")
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        if isinstance(snapshot_chars, bool) or not isinstance(snapshot_chars, int):
            raise TypeError("snapshot_chars must be an integer")
        if snapshot_chars < 1 or snapshot_chars > MAX_SNAPSHOT_CHARS:
            raise ValueError(f"snapshot_chars must be within [1, {MAX_SNAPSHOT_CHARS}]")

        basename = _normalised_basename(filename)
        name = (basename or "document")[:MAX_FILENAME_CHARS]
        extension = Path(basename).suffix.lower()[:MAX_EXTENSION_CHARS]
        mime_type = (mimetypes.guess_type(basename)[0] or "application/octet-stream")[:MAX_MIME_CHARS]
        return cls(
            filename=name,
            extension=extension,
            mime_type=mime_type,
            size_bytes=len(file_bytes),
            text_snapshot=_bounded_text(text, snapshot_chars),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the metadata-only payload sent alongside ``text_snapshot``."""
        return {
            "filename": self.filename,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }

    def to_state(self) -> OrchidDecisionState:
        """Render the descriptor as decision state.

        The bounded snapshot is the state ``text``; all other fields live in the
        JSON-safe payload. No full document body is included.
        """
        return OrchidDecisionState(
            text=self.text_snapshot,
            payload={"document": self.to_payload()},
        )


@dataclass(frozen=True)
class DocumentSelection:
    """The outcome of one parser or strategy selection.

    ``source`` distinguishes a confident provider answer (``decision``) from an
    operator-narrowed single-entry allowlist (``configured``), an operational
    fallback (``fallback``) and an unconfigured/disabled point (``disabled``).
    Only ``decision`` and ``configured`` carry a usable ``value``.
    """

    value: str | None
    source: Literal["decision", "configured", "fallback", "disabled"]
    confidence: float | None = None
    fallback_reason: str | None = None
    provider: str = ""

    @property
    def accepted(self) -> bool:
        """Whether a confident or configured registered name was selected."""
        return self.source in ("decision", "configured") and self.value is not None


async def select_document_parser(
    descriptor: DocumentDescriptor,
    binding: OrchidDecisionPointBinding | None,
    *,
    timeout: float | None = None,
) -> DocumentSelection:
    """Ask the ``ingestion`` point which registered parser to use.

    ``binding=None`` means the point is disabled, unconfigured, or its provider
    was not built: no provider call and no decision line. Every operational
    failure falls back to the extension-based parser owned by the caller.
    """
    return await _select(
        descriptor,
        binding,
        candidates=list_parser_names(),
        option_key="parsers",
        question=_PARSER_QUESTION,
        instructions=_PARSER_INSTRUCTIONS,
        unknown_reason="unknown_parser",
        timeout=timeout,
    )


async def select_ingestion_strategy(
    descriptor: DocumentDescriptor,
    binding: OrchidDecisionPointBinding | None,
    *,
    timeout: float | None = None,
) -> DocumentSelection:
    """Ask the ``ingestion`` point which registered strategy to use.

    Called after text extraction and before ``strategy.ingest``. ``binding=None``
    keeps the configured (or default) strategy with zero provider calls; every
    failure falls back identically.
    """
    return await _select(
        descriptor,
        binding,
        candidates=tuple(sorted(INGESTION_REGISTRY)),
        option_key="strategies",
        question=_STRATEGY_QUESTION,
        instructions=_STRATEGY_INSTRUCTIONS,
        unknown_reason="unknown_strategy",
        timeout=timeout,
    )


async def _select(
    descriptor: DocumentDescriptor,
    binding: OrchidDecisionPointBinding | None,
    *,
    candidates: tuple[str, ...],
    option_key: str,
    question: str,
    instructions: str,
    unknown_reason: str,
    timeout: float | None,
) -> DocumentSelection:
    if binding is None:
        return DocumentSelection(value=None, source="disabled")

    selected_candidates, allowlist_active = _apply_allowlist(binding, candidates, option_key)
    if selected_candidates is None:
        return _fallback(
            binding,
            "invalid_options",
            detail=f"options.{option_key} must list registered names",
        )

    limit = binding.max_candidates or MAX_CANDIDATES
    if len(selected_candidates) > limit:
        return _fallback(binding, "too_many_candidates")
    if allowlist_active and len(selected_candidates) == 1:
        # A single-entry allowlist is an operator policy, not a decision: use it
        # without a provider call so narrowing the candidates cannot accidentally
        # fall back to a name outside the allowlist.
        configured = selected_candidates[0]
        logger.info("[Ingestion] single-entry allowlist selected %s=%r", question, configured)
        return DocumentSelection(value=configured, source="configured")
    if len(selected_candidates) < 2:
        return _fallback(binding, "no_candidates")

    decision_question = OrchidDecisionQuestion(
        name=question,
        kind="choice",
        options=selected_candidates,
        instructions=instructions,
    )
    started = time.perf_counter()
    try:
        result = await binding.decide(descriptor.to_state(), [decision_question], timeout=timeout)
    except OrchidDecisionError as exc:
        logger.warning("[Ingestion] %s decision provider failed code=%s", question, exc.code)
        return _fallback(
            binding,
            exc.code or "provider_error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=exc.provider,
        )
    except TimeoutError:
        return _fallback(
            binding,
            "timeout",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        # Only the exception type crosses this seam: provider messages can echo
        # document content, and a traceback would duplicate it in logs.
        logger.warning("[Ingestion] %s decision provider raised %s", question, type(exc).__name__)
        return _fallback(
            binding,
            "provider_error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        result_provider = result.provider
        answers = result.answers
    except Exception:
        return _fallback(binding, "invalid_response", latency_ms=latency_ms)

    provider = result_provider if isinstance(result_provider, str) and result_provider else binding.provider
    if not isinstance(answers, Mapping):
        return _fallback(binding, "invalid_response", latency_ms=latency_ms, provider=provider)
    answer = answers.get(question)
    if not isinstance(answer, OrchidDecisionAnswer):
        return _fallback(binding, "invalid_response", latency_ms=latency_ms, provider=provider)
    if answer.refusal is not None:
        return _fallback(
            binding,
            "refusal",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.kind != "choice":
        return _fallback(
            binding,
            "invalid_response",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.confidence < binding.confidence_threshold:
        return _fallback(
            binding,
            "low_confidence",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    selected = str(answer.value)
    if selected not in selected_candidates:
        return _fallback(
            binding,
            "out_of_allowlist" if allowlist_active else unknown_reason,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    log_decision(
        point=binding.point,
        path="decision",
        provider=provider,
        latency_ms=latency_ms,
        confidence=answer.confidence,
        outcome="accepted",
    )
    logger.info("[Ingestion] decision selected %s=%r", question, selected)
    return DocumentSelection(
        value=selected,
        source="decision",
        confidence=answer.confidence,
        provider=provider,
    )


def _apply_allowlist(
    binding: OrchidDecisionPointBinding,
    candidates: tuple[str, ...],
    option_key: str,
) -> tuple[tuple[str, ...] | None, bool]:
    """Narrow ``candidates`` to the operator allowlist, if declared.

    ``None`` signals an invalid declaration (malformed, empty, or containing an
    unregistered name). The caller falls back without a provider call so a typo
    cannot silently shrink or invent the candidate set.
    """
    raw = binding.options.get(option_key)
    if raw is None:
        return candidates, False
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or not raw:
        return None, False
    if any(not isinstance(name, str) or not name.strip() for name in raw):
        return None, False
    allowed = {name for name in raw}
    if any(name not in candidates for name in allowed):
        return None, False
    narrowed = tuple(name for name in candidates if name in allowed)
    return narrowed, True


def _fallback(
    binding: OrchidDecisionPointBinding,
    reason: str,
    *,
    latency_ms: float | None = None,
    confidence: float | None = None,
    provider: str = "",
    detail: str = "",
) -> DocumentSelection:
    """Emit one structured fallback line and let the caller use the default."""
    message = reason if not detail else f"{reason} ({detail})"
    logger.warning("[Ingestion] decision fell back to the configured default (reason=%s)", message)
    log_decision(
        point=binding.point,
        path="llm_fallback",
        provider=provider or binding.provider,
        latency_ms=latency_ms,
        confidence=confidence,
        outcome="fallback",
        fallback_reason=reason,
    )
    return DocumentSelection(
        value=None,
        source="fallback",
        confidence=confidence,
        fallback_reason=reason,
        provider=provider or binding.provider,
    )


def _normalised_basename(value: str) -> str:
    """Strip directories and control characters without truncating."""
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    basename = _CONTROL_CHARS_RE.sub(" ", basename).strip()
    return " ".join(basename.split())


def _bounded_text(value: str | None, limit: int) -> str:
    """Collapse a bounded prefix of ``value`` into a ≤``limit`` snapshot.

    Only the first ``limit * 4`` characters are scanned; the rest of a large
    document is never copied or traversed. Whitespace collapsing can therefore
    shrink the snapshot below ``limit`` when the prefix is mostly whitespace.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("text must be a string or None")
    prefix = value[: limit * 4]
    text = _CONTROL_CHARS_RE.sub(" ", prefix)
    text = " ".join(text.split())
    return text[:limit]
