"""
Decision-backed guardrail — delegate allow/block/redact/warn to a decision point.

The rail implements the existing async :class:`~orchid_ai.core.guardrails.OrchidGuardrail`
contract, so it composes with the deterministic rails in the same chain. It is
opt-in: a rail without a resolved ``guardrails`` binding never calls a provider
and applies its failure policy instead.

The content sent to the provider is bounded and redacted locally. Redaction is
best-effort and deliberately conservative:

* PII — emails, phone numbers, card numbers, SSNs (dash- or space-separated),
  IPv4 addresses.
* Credentials — bearer/Basic/Digest tokens, bare JWTs, AWS access keys,
  Google API keys, common provider-prefixed API keys, private-key blocks.
* Assignments and prose — ``token=``/``secret_key:`` style pairs and
  ``password is <value>`` phrasing.

Unknown secret shapes can still evade the rules. The state is truncated as a
second boundary; when that happens no verdict is accepted (see below).

Provider refusals, low confidence, timeouts, errors, and truncated input apply
the rail's ``on_failure`` policy (``open`` by default in code; configuration
requires an explicit mode for ``type: decision``). Provider text is never
reused as the user-facing message. A ``redact`` verdict replaces the whole
content with a safe placeholder because no span-level protocol exists.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Final

from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailContext,
    OrchidGuardrailFailurePolicy,
    OrchidGuardrailFailurePolicyAware,
    OrchidGuardrailResult,
)
from ..decision.observability import log_decision, safe_log_token

logger = logging.getLogger(__name__)

__all__ = ["DecisionGuardrail"]

_POINT: Final = "guardrails"
_VERDICT_QUESTION: Final = "verdict"
_VERDICTS: Final = ("allow", "block", "redact", "warn", "log")
_MAX_STATE_TEXT_CHARS: Final = 2000
_MAX_ALLOWED_STATE_TEXT_CHARS: Final = 100_000
_MAX_DECISION_SECONDS: Final = 2.0
_MAX_ALLOWED_TIMEOUT_SECONDS: Final = 10.0
_MAX_AGENT_NAME_CHARS: Final = 64
_DEFAULT_BLOCK_MESSAGE: Final = "This content was blocked by a policy check."
_DEFAULT_WARN_MESSAGE: Final = "This content may require review."
_DEFAULT_REDACTION_PLACEHOLDER: Final = "[REDACTED]"

# ── Local redaction ──────────────────────────────────────────
# Applied before any provider call. The patterns are intentionally conservative:
# a false positive removes content the provider never needed, while a false
# negative would ship a secret to a third party.

_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        # Complete PEM blocks first: they can contain assignment-like lines.
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        # An unterminated key block: drop everything after the BEGIN marker.
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
        "[REDACTED_JWT]",
    ),
    (
        re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b"),
        "[REDACTED_AWS_ACCESS_KEY]",
    ),
    (
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "[REDACTED_GOOGLE_API_KEY]",
    ),
    (
        # Base64 payloads usually mix case; requiring both avoids matching
        # prose such as "basic authentication".
        re.compile(
            r"\b(?:[Bb]asic|[Dd]igest)\s+(?=[A-Za-z0-9+/=]{8,}\b)(?=[^ \n]*[A-Z])(?=[^ \n]*[a-z])[A-Za-z0-9+/=]{8,}"
        ),
        "[REDACTED_AUTH_HEADER]",
    ),
    (
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
        "[REDACTED_BEARER_TOKEN]",
    ),
    (
        re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|github_pat|glpat|xapp|xox[abprs])[-_][A-Za-z0-9_-]{12,}\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        # ``token=...`` / ``secret_key: ...`` style assignments.
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token"
            r"|token|secret[_-]?key|client[_-]?secret|private[_-]?key|secret|password|passwd)"
            r"\s*[:=]\s*\S{4,}"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        # Prose form: "the password is hunter2".
        re.compile(r"(?i)\b(?:password|passphrase|secret|token)\s+is\s+\S{4,}"),
        "[REDACTED_SECRET]",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        "[REDACTED_CARD]",
    ),
    (
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[REDACTED_IP]",
    ),
)


def _redact_text(content: str) -> tuple[str, int]:
    """Return ``content`` with known secret/PII shapes replaced, plus a count."""
    redacted = content
    replacements = 0
    for pattern, replacement in _REDACTION_RULES:
        redacted, count = pattern.subn(replacement, redacted)
        replacements += count
    return redacted, replacements


def _bounded_state_text(content: str, limit: int) -> tuple[str, int, bool]:
    """Redact then truncate before the text crosses the provider boundary."""
    redacted, replacements = _redact_text(content)
    return redacted[:limit], replacements, len(redacted) > limit


def _safe_agent_name(value: str) -> str:
    return (value or "").strip()[:_MAX_AGENT_NAME_CHARS]


def _direction_value(context: OrchidGuardrailContext) -> str:
    direction = context.direction
    return str(getattr(direction, "value", direction))


def _instructions(direction: str) -> str:
    return (
        f"This is {direction} content for an assistant. Choose 'allow' when it may pass unchanged, "
        "'block' when it must not pass, 'redact' when sensitive details must be removed but the "
        "remaining content may pass, 'warn' when it may pass with a user-visible warning, and "
        "'log' when it may pass while the decision is recorded."
    )


def _accepted_details(verdict: str, confidence: float, provider: str) -> dict[str, object]:
    return {
        "decision": "accepted",
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "provider": safe_log_token(provider) if provider else "",
    }


def _fallback_details(reason: str, on_failure: OrchidGuardrailFailurePolicy) -> dict[str, object]:
    return {
        "decision": "fallback",
        "fallback_reason": reason,
        "on_failure": on_failure.value,
    }


class DecisionGuardrail(OrchidGuardrail, OrchidGuardrailFailurePolicyAware):
    """
    Ask the ``guardrails`` decision point how to treat one piece of content.

    Parameters
    ----------
    binding : OrchidDecisionPointBinding | None
        Resolved ``guardrails`` point. ``None`` means no provider is configured
        for this scope; the rail then applies ``on_failure`` without a call.
    fail_action : str
        Action used when ``on_failure`` is ``closed``. One of ``block``,
        ``redact``, ``warn``, or ``log``. Default: ``block``.
    on_failure : str
        ``open`` lets content pass when no decision is available; ``closed``
        applies ``fail_action``. The rail constructor defaults to ``open`` for
        programmatic use, but ``type: decision`` configuration must state the
        mode explicitly (ADR-002).
    configuration_reason : str
        Stable reason recorded when ``binding`` is ``None``: ``not_configured``,
        ``disabled``, or ``provider_missing``. Supplied by the graph wiring.
    point : str
        Must be ``guardrails``. Declared explicitly so a mis-wired rail fails
        at construction instead of querying the wrong decision point.
    timeout_seconds : float
        Per-check wall-clock ceiling. Default: 2 seconds, maximum 10.
    max_state_chars : int
        Maximum redacted characters sent to the provider. Default: 2000. When
        the redacted text exceeds it, no provider call is made and
        ``on_failure`` applies.
    block_message : str
        Safe, generic user-facing message for a ``block`` verdict.
    warn_message : str
        Safe, generic user-facing message for a ``warn`` verdict.
    redaction_placeholder : str
        Whole-content replacement for a ``redact`` verdict. There is no
        span-level protocol, so the original content is never reused.
    """

    def __init__(
        self,
        *,
        binding: OrchidDecisionPointBinding | None = None,
        fail_action: str = "block",
        on_failure: str | OrchidGuardrailFailurePolicy = OrchidGuardrailFailurePolicy.OPEN,
        configuration_reason: str = "not_configured",
        point: str = _POINT,
        timeout_seconds: float = _MAX_DECISION_SECONDS,
        max_state_chars: int = _MAX_STATE_TEXT_CHARS,
        block_message: str = _DEFAULT_BLOCK_MESSAGE,
        warn_message: str = _DEFAULT_WARN_MESSAGE,
        redaction_placeholder: str = _DEFAULT_REDACTION_PLACEHOLDER,
    ) -> None:
        self._binding = binding
        self._configuration_reason = _validate_configuration_reason(configuration_reason)
        self._warned_reasons: set[str] = set()

        if point != _POINT:
            raise ValueError(f"decision guardrail must use the {_POINT!r} point, got {point!r}")

        fail_action_value = OrchidGuardrailAction(fail_action)
        if fail_action_value is OrchidGuardrailAction.ALLOW:
            raise ValueError("fail_action must be one of 'block', 'redact', 'warn', or 'log'")
        self._fail_action = fail_action_value

        if isinstance(on_failure, OrchidGuardrailFailurePolicy):
            self._on_failure = on_failure
        else:
            self._on_failure = OrchidGuardrailFailurePolicy(on_failure)

        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._max_state_chars = _validate_max_state_chars(max_state_chars)
        self._block_message = _validate_message(block_message, "block_message")
        self._warn_message = _validate_message(warn_message, "warn_message")
        self._redaction_placeholder = _validate_message(redaction_placeholder, "redaction_placeholder")

    # ── Contract ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "decision"

    @property
    def point(self) -> str:
        """The decision point this rail consumes."""
        return _POINT

    @property
    def on_failure(self) -> OrchidGuardrailFailurePolicy:
        """Configured fail-open or fail-closed policy."""
        return self._on_failure

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        binding = self._binding
        if binding is None:
            return self._fallback(self._configuration_reason, started=None)

        direction = _direction_value(context)
        bounded_text, redactions, truncated = _bounded_state_text(content, self._max_state_chars)
        if truncated:
            # A prefix is not a verdict: padding must not smuggle harmful
            # content past the decision. Open lets it pass; closed applies the
            # rail's fail action. Raise max_state_chars for long-form input.
            return self._fallback("truncated", started=None, provider=binding.provider)

        state = OrchidDecisionState(
            text=bounded_text,
            payload={
                "direction": direction,
                "agent": _safe_agent_name(context.agent_name),
                "content_chars": len(content),
                "redactions": redactions,
                "truncated": False,
            },
        )
        question = OrchidDecisionQuestion(
            name=_VERDICT_QUESTION,
            kind="choice",
            options=_VERDICTS,
            instructions=_instructions(direction),
        )

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await binding.decide(state, [question], timeout=self._timeout_seconds)
        except OrchidDecisionError as exc:
            return self._fallback(
                exc.code or "provider_error",
                started=started,
                provider=exc.provider or binding.provider,
            )
        except TimeoutError:
            return self._fallback("timeout", started=started, provider=binding.provider)
        except Exception as exc:
            # Only the exception type crosses this seam: provider messages can
            # carry user state and must not reach the logs.
            logger.debug("[Guardrails] decision provider raised %s", type(exc).__name__)
            return self._fallback("provider_error", started=started, provider=binding.provider)

        latency_ms = (time.perf_counter() - started) * 1000.0
        provider = result.provider or binding.provider

        answer = result.answers.get(_VERDICT_QUESTION)
        if answer is None:
            return self._fallback("invalid_response", started=started, provider=provider)
        if answer.refusal is not None:
            return self._fallback(
                "refusal",
                started=started,
                provider=provider,
                confidence=answer.confidence,
            )
        if answer.kind != "choice" or not isinstance(answer.value, str) or answer.value not in _VERDICTS:
            return self._fallback(
                "invalid_response",
                started=started,
                provider=provider,
                confidence=answer.confidence,
            )
        if answer.confidence < binding.confidence_threshold:
            return self._fallback(
                "low_confidence",
                started=started,
                provider=provider,
                confidence=answer.confidence,
            )

        log_decision(
            point=_POINT,
            path="decision",
            provider=provider,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            outcome="accepted",
        )
        return self._accepted_result(answer.value, answer.confidence, provider)

    # ── Result mapping ──────────────────────────────────────

    def _accepted_result(
        self,
        verdict: str,
        confidence: float,
        provider: str,
    ) -> OrchidGuardrailResult:
        """Map a confident verdict onto the existing guardrail result contract."""
        details = _accepted_details(verdict, confidence, provider)

        if verdict == "allow":
            return OrchidGuardrailResult(triggered=False, guardrail_name=self.name, details=details)

        if verdict == "block":
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.BLOCK,
                guardrail_name=self.name,
                message=self._block_message,
                details=details,
            )

        if verdict == "redact":
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.REDACT,
                guardrail_name=self.name,
                message="Sensitive details were removed by a policy check.",
                redacted_content=self._redaction_placeholder,
                details=details,
            )

        if verdict == "warn":
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.WARN,
                guardrail_name=self.name,
                message=self._warn_message,
                details=details,
            )

        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.LOG,
            guardrail_name=self.name,
            details=details,
        )

    def _fallback(
        self,
        reason: str,
        *,
        started: float | None,
        provider: str = "",
        confidence: float | None = None,
    ) -> OrchidGuardrailResult:
        """Apply the configured failure policy after one structured log line."""
        log_decision(
            point=_POINT,
            path="llm_fallback",
            provider=provider,
            latency_ms=None if started is None else (time.perf_counter() - started) * 1000.0,
            confidence=confidence,
            outcome="fallback",
            fallback_reason=reason,
        )
        self._warn_once(reason)

        if self._on_failure is OrchidGuardrailFailurePolicy.OPEN:
            return OrchidGuardrailResult(
                triggered=False,
                guardrail_name=self.name,
                details=_fallback_details(reason, self._on_failure),
            )

        details = _fallback_details(reason, self._on_failure)

        if self._fail_action is OrchidGuardrailAction.BLOCK:
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.BLOCK,
                guardrail_name=self.name,
                message=self._block_message,
                details=details,
            )

        if self._fail_action is OrchidGuardrailAction.REDACT:
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.REDACT,
                guardrail_name=self.name,
                message="Sensitive details were removed by a policy check.",
                redacted_content=self._redaction_placeholder,
                details=details,
            )

        if self._fail_action is OrchidGuardrailAction.WARN:
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.WARN,
                guardrail_name=self.name,
                message=self._warn_message,
                details=details,
            )

        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.LOG,
            guardrail_name=self.name,
            details=details,
        )

    def _warn_once(self, reason: str) -> None:
        """Log a fallback warning once per rail scope, not once per check.

        The structured decision line still records every fallback; the human
        warning only announces the first occurrence of a reason in this scope.
        """
        if reason in self._warned_reasons:
            logger.debug(
                "[Guardrails] decision rail %s fallback repeated (reason=%s)",
                self._on_failure.value,
                reason,
            )
            return
        self._warned_reasons.add(reason)
        logger.warning(
            "[Guardrails] decision rail %s fallback (reason=%s); further occurrences are logged at debug",
            self._on_failure.value,
            reason,
        )


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    timeout = float(value)
    if not 0.0 < timeout <= _MAX_ALLOWED_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be within (0, {_MAX_ALLOWED_TIMEOUT_SECONDS}]")
    return timeout


def _validate_max_state_chars(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_state_chars must be an integer")
    if not 1 <= value <= _MAX_ALLOWED_STATE_TEXT_CHARS:
        raise ValueError(f"max_state_chars must be within [1, {_MAX_ALLOWED_STATE_TEXT_CHARS}]")
    return value


def _validate_message(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _validate_configuration_reason(value: str) -> str:
    """Allow an empty reason when a binding was resolved successfully."""
    if not isinstance(value, str):
        raise TypeError("configuration_reason must be a string")
    return value.strip()
