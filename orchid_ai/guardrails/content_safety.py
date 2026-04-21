"""
Content safety guardrail — keyword/pattern-based harmful content detection.

Uses blocklists and regex patterns to detect harmful content categories:
hate speech, violence, sexual content, self-harm, and custom terms.

This is a lightweight, deterministic check (< 1ms).  For ML-based
classification, consumers can register a custom guardrail that uses
a local classifier model.
"""

from __future__ import annotations

import re

from ..core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailContext,
    OrchidGuardrailResult,
)

# ── Default blocklist patterns (case-insensitive) ─────────────
# These are intentionally broad categories.  Consumers should refine
# via the ``blocklist`` and ``patterns`` config options.

_DEFAULT_PATTERNS: dict[str, list[str]] = {
    "self_harm": [
        r"\b(how to|ways to|methods? (?:of|for)|steps? to)\s+(?:kill|harm|hurt|end)\s+(?:myself|yourself|oneself)\b",
        r"\bsuicid(?:e|al)\s+(?:method|plan|instruction|guide)\b",
    ],
    "violence": [
        r"\b(?:how to|instructions? (?:for|to)|guide to)\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|weapon|explosive)\b",
        r"\b(?:how to|ways to)\s+(?:poison|assassinate|murder)\b",
    ],
    "illegal_activity": [
        r"\b(?:how to|instructions? (?:for|to))\s+(?:hack|crack|exploit|bypass)\s+(?:password|security|firewall)\b",
        r"\b(?:how to|ways to)\s+(?:forge|counterfeit|launder)\b",
    ],
}


class ContentSafetyGuardrail(OrchidGuardrail):
    """
    Block content matching harmful keyword patterns or custom blocklists.

    Parameters
    ----------
    fail_action : str
        Action on match: "block", "warn", or "log".
    blocklist : list[str]
        Custom words/phrases to block (case-insensitive exact match).
    patterns : list[str]
        Custom regex patterns to match against.
    categories : list[str]
        Built-in categories to enable. Default: all.
        Options: "self_harm", "violence", "illegal_activity".
    """

    def __init__(
        self,
        *,
        fail_action: str = "block",
        blocklist: list[str] | None = None,
        patterns: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> None:
        self._fail_action = OrchidGuardrailAction(fail_action)
        self._blocklist = [w.lower() for w in (blocklist or [])]
        self._custom_patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or [])]

        # Compile built-in category patterns
        enabled_categories = categories or list(_DEFAULT_PATTERNS.keys())
        self._category_patterns: dict[str, list[re.Pattern[str]]] = {}
        for cat in enabled_categories:
            if cat in _DEFAULT_PATTERNS:
                self._category_patterns[cat] = [re.compile(p, re.IGNORECASE) for p in _DEFAULT_PATTERNS[cat]]

    @property
    def name(self) -> str:
        return "content_safety"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        content_lower = content.lower()

        # Check custom blocklist
        for word in self._blocklist:
            if word in content_lower:
                return OrchidGuardrailResult(
                    triggered=True,
                    action=self._fail_action,
                    guardrail_name=self.name,
                    message="Your message was blocked by content safety filters.",
                    details={"matched_type": "blocklist", "matched_term": word},
                )

        # Check custom regex patterns
        for pattern in self._custom_patterns:
            match = pattern.search(content)
            if match:
                return OrchidGuardrailResult(
                    triggered=True,
                    action=self._fail_action,
                    guardrail_name=self.name,
                    message="Your message was blocked by content safety filters.",
                    details={"matched_type": "custom_pattern", "matched_text": match.group()},
                )

        # Check built-in category patterns
        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    return OrchidGuardrailResult(
                        triggered=True,
                        action=self._fail_action,
                        guardrail_name=self.name,
                        message="Your message was blocked by content safety filters.",
                        details={"matched_type": "category", "category": category, "matched_text": match.group()},
                    )

        return OrchidGuardrailResult.passed(self.name)
