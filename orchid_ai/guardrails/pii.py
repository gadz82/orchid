"""
PII detection guardrail — regex-based detection and redaction of
personally identifiable information.

Detects:
  - Email addresses
  - Phone numbers (international formats)
  - Credit card numbers (Visa, Mastercard, Amex, Discover)
  - Social Security Numbers (US SSN)
  - IP addresses (IPv4)

When ``fail_action`` is ``"redact"``, matched PII is replaced with
``[REDACTED_<TYPE>]`` placeholders.

This is a pure regex implementation with zero external dependencies.
For production-grade PII detection, consumers can register a custom
guardrail backed by Microsoft Presidio or a local NER model.
"""

from __future__ import annotations

import re

from ..core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
)

# ── PII patterns ─────────────────────────────────────────────

_PII_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "email": (
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    "phone": (
        re.compile(
            r"(?<!\d)"  # not preceded by digit
            r"(?:\+?1[-.\s]?)?"  # optional country code
            r"(?:\(?\d{3}\)?[-.\s]?)"  # area code
            r"\d{3}[-.\s]?\d{4}"  # number
            r"(?!\d)"  # not followed by digit
        ),
        "[REDACTED_PHONE]",
    ),
    "credit_card": (
        re.compile(
            r"\b(?:"
            r"4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # Visa
            r"|5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # Mastercard
            r"|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}"  # Amex
            r"|6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # Discover
            r")\b"
        ),
        "[REDACTED_CREDIT_CARD]",
    ),
    "ssn": (
        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    "ipv4": (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[REDACTED_IP]",
    ),
}


class PIIDetectionGuardrail(Guardrail):
    """
    Detect and optionally redact PII in content.

    Parameters
    ----------
    fail_action : str
        "block" to reject the message, "redact" to replace PII with
        placeholders, "warn" to allow but flag, "log" to silently log.
    entities : list[str]
        PII types to detect.  Default: all available.
        Options: "email", "phone", "credit_card", "ssn", "ipv4".
    """

    def __init__(
        self,
        *,
        fail_action: str = "redact",
        entities: list[str] | None = None,
    ) -> None:
        self._fail_action = GuardrailAction(fail_action)

        enabled = entities or list(_PII_PATTERNS.keys())
        self._patterns: dict[str, tuple[re.Pattern[str], str]] = {
            name: _PII_PATTERNS[name] for name in enabled if name in _PII_PATTERNS
        }

    @property
    def name(self) -> str:
        return "pii_detection"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        found_entities: list[dict[str, str]] = []
        redacted = content

        for entity_type, (pattern, replacement) in self._patterns.items():
            matches = pattern.findall(content)
            if matches:
                for match_text in matches:
                    found_entities.append({"type": entity_type, "value": match_text})
                redacted = pattern.sub(replacement, redacted)

        if not found_entities:
            return GuardrailResult.passed(self.name)

        entity_types = list({e["type"] for e in found_entities})

        return GuardrailResult(
            triggered=True,
            action=self._fail_action,
            guardrail_name=self.name,
            message=(
                f"Personal information detected ({', '.join(entity_types)}). "
                "Content has been filtered for privacy protection."
            ),
            redacted_content=redacted if self._fail_action == GuardrailAction.REDACT else None,
            details={
                "entities_found": len(found_entities),
                "entity_types": entity_types,
            },
        )
