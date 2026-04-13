"""
Topic restriction guardrail — enforce per-agent domain boundaries.

Uses keyword matching to determine if a user's query is on-topic for
a given agent.  Each agent can specify ``allowed_topics`` — a list of
keywords/phrases that define its domain.

For more sophisticated topic classification, consumers can register a
custom guardrail backed by an embedding similarity model or LLM classifier.
"""

from __future__ import annotations

from ..core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
)


class TopicRestrictionGuardrail(Guardrail):
    """
    Block queries that don't match any of the allowed topics.

    Parameters
    ----------
    fail_action : str
        Action when off-topic: "block", "warn", or "log".
    allowed_topics : list[str]
        Keywords/phrases that define on-topic content.
        The query is on-topic if ANY allowed topic appears in it
        (case-insensitive substring match).
    threshold : float
        Reserved for future use with embedding-based matching.
        Currently unused (keyword matching is binary).
    """

    def __init__(
        self,
        *,
        fail_action: str = "block",
        allowed_topics: list[str] | None = None,
        threshold: float = 0.7,
    ) -> None:
        self._fail_action = GuardrailAction(fail_action)
        self._allowed_topics = [t.lower() for t in (allowed_topics or [])]
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "topic_restriction"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        if not self._allowed_topics:
            # No restrictions configured — allow everything
            return GuardrailResult.passed(self.name)

        content_lower = content.lower()

        for topic in self._allowed_topics:
            if topic in content_lower:
                return GuardrailResult.passed(self.name)

        agent_label = f" for the {context.agent_name} agent" if context.agent_name else ""
        return GuardrailResult(
            triggered=True,
            action=self._fail_action,
            guardrail_name=self.name,
            message=(
                f"Your question appears to be outside the scope of topics handled{agent_label}. "
                f"Supported topics include: {', '.join(self._allowed_topics[:5])}."
            ),
            details={
                "allowed_topics": self._allowed_topics,
                "agent_name": context.agent_name,
            },
        )
