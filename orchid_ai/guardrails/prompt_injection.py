"""
Prompt injection guardrail — detect adversarial inputs that attempt
to override system instructions.

Uses pattern matching to detect common injection techniques:
  - Direct instruction override ("ignore previous instructions")
  - Role-play attacks ("you are now DAN")
  - System prompt extraction ("repeat your system prompt")
  - Delimiter injection ("```system", "<|im_start|>")

This is a lightweight, deterministic check (< 1ms).
"""

from __future__ import annotations

import re

from ..core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
)

# ── Default injection patterns (case-insensitive) ─────────────

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Direct instruction override
    (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|guidelines?)",
        "instruction_override",
    ),
    (
        r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
        "instruction_override",
    ),
    (
        r"forget\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
        "instruction_override",
    ),
    (r"(?:do\s+not|don'?t)\s+follow\s+(?:your|the|any)\s+(?:instructions?|rules?|guidelines?)", "instruction_override"),
    (r"override\s+(?:your|the|system)\s+(?:instructions?|prompt|rules?|settings?)", "instruction_override"),
    # Role-play / persona hijack
    (r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|unrestricted|jailbroken|unfiltered)", "persona_hijack"),
    (
        r"(?:pretend|act|behave)\s+(?:like|as\s+if)\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|limits|filters)",
        "persona_hijack",
    ),
    (r"enter\s+(?:developer|debug|admin|god|unrestricted)\s+mode", "persona_hijack"),
    (r"switch\s+to\s+(?:developer|debug|jailbreak|unrestricted)\s+mode", "persona_hijack"),
    # System prompt extraction
    (
        r"(?:repeat|show|display|print|output|reveal|tell\s+me)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
        "prompt_extraction",
    ),
    (r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?|initial\s+prompt)", "prompt_extraction"),
    (r"(?:copy|paste|echo)\s+(?:the\s+)?(?:above|system|initial)\s+(?:prompt|text|instructions?)", "prompt_extraction"),
    # Delimiter injection
    (r"<\|(?:im_start|im_end|system|endoftext)\|>", "delimiter_injection"),
    (r"```\s*system\s*\n", "delimiter_injection"),
    (r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]", "delimiter_injection"),
]


class PromptInjectionGuardrail(Guardrail):
    """
    Detect and block prompt injection attempts.

    Parameters
    ----------
    fail_action : str
        Action on detection: "block" (default), "warn", or "log".
    extra_patterns : list[str]
        Additional regex patterns to check (case-insensitive).
    """

    def __init__(
        self,
        *,
        fail_action: str = "block",
        extra_patterns: list[str] | None = None,
    ) -> None:
        self._fail_action = GuardrailAction(fail_action)

        # Compile all patterns
        self._patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pattern, re.IGNORECASE), category) for pattern, category in _INJECTION_PATTERNS
        ]

        # Add custom patterns
        for pattern in extra_patterns or []:
            self._patterns.append((re.compile(pattern, re.IGNORECASE), "custom"))

    @property
    def name(self) -> str:
        return "prompt_injection"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        for pattern, category in self._patterns:
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    triggered=True,
                    action=self._fail_action,
                    guardrail_name=self.name,
                    message="Your message was flagged as a potential prompt injection and has been blocked.",
                    details={
                        "category": category,
                        "matched_text": match.group(),
                    },
                )

        return GuardrailResult.passed(self.name)
