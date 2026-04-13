"""
Max-length guardrail — blocks content exceeding a character or token limit.

This is the simplest guardrail: pure deterministic check, zero latency.
"""

from __future__ import annotations

from ..core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
)


class MaxLengthGuardrail(Guardrail):
    """Block or warn when content exceeds a maximum character count."""

    def __init__(
        self,
        *,
        fail_action: str = "block",
        max_characters: int = 10000,
    ) -> None:
        self._fail_action = GuardrailAction(fail_action)
        self._max_characters = max_characters

    @property
    def name(self) -> str:
        return "max_length"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        length = len(content)
        if length <= self._max_characters:
            return GuardrailResult.passed(self.name)

        return GuardrailResult(
            triggered=True,
            action=self._fail_action,
            guardrail_name=self.name,
            message=(
                f"Content exceeds the maximum allowed length "
                f"({length:,} characters, limit is {self._max_characters:,})."
            ),
            details={"length": length, "limit": self._max_characters},
        )
