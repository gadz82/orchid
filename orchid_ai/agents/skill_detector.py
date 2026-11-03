"""Skill detection — matches user queries to agent-level skills.

A configured decision provider is tried first; the chat-model prompt remains
the fallback when no provider is configured, the answer is low-confidence, or
the provider fails.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config.schema import OrchidAgentSkillConfig
from ..core.decision import (
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionState,
    require_confidence,
)

logger = logging.getLogger(__name__)

_SKILL_NONE = "__none__"


class SkillDetector:
    """Detects whether a user query matches an agent-level skill."""

    def __init__(
        self,
        chat_model: Any | None = None,
        *,
        decision_model: OrchidDecisionModel | None = None,
        confidence_threshold: float = 0.8,
        provider_name: str = "",
    ) -> None:
        self._chat_model = chat_model
        self._decision_model = decision_model
        self._threshold = max(0.0, min(1.0, confidence_threshold))
        self._provider_name = provider_name

    async def detect(self, query: str, skills: dict[str, OrchidAgentSkillConfig]) -> str | None:
        """Return the matching skill name, or None."""
        if not skills:
            return None

        if self._decision_model is not None:
            decided = await self._detect_by_decision(query, skills)
            if decided is not _NO_DECISION:
                return decided

        if self._chat_model is None:
            return None
        return await self._detect_by_llm(query, skills)

    async def _detect_by_decision(
        self,
        query: str,
        skills: dict[str, OrchidAgentSkillConfig],
    ) -> str | None:
        descriptions = "\n".join(f"- {name}: {skill.description}" for name, skill in skills.items())
        question = OrchidDecisionQuestion(
            name="skill",
            kind="choice",
            options=(_SKILL_NONE, *skills.keys()),
            instructions=(
                "Does the user query match one of these skills? "
                f"Choose {_SKILL_NONE!r} when none matches.\n{descriptions}"
            ),
        )
        try:
            assert self._decision_model is not None
            result = await self._decision_model.decide(
                OrchidDecisionState(text=query[:4000]),
                [question],
            )
        except Exception:
            logger.warning("[SkillDetector] decision provider failed; falling back to the LLM", exc_info=True)
            return _NO_DECISION

        answer = require_confidence(result, "skill", self._threshold)
        if answer is None:
            return _NO_DECISION
        return None if answer.value == _SKILL_NONE else str(answer.value)

    async def _detect_by_llm(
        self,
        query: str,
        skills: dict[str, OrchidAgentSkillConfig],
    ) -> str | None:
        skill_descriptions = "\n".join(f'- "{name}": {skill.description}' for name, skill in skills.items())

        prompt = (
            f"User query: {query}\n\n"
            f"Available skills for this agent:\n{skill_descriptions}\n\n"
            "If the user's query closely matches one of these skills, respond with "
            'ONLY the skill name (e.g. "course_completion_summary").\n'
            'If no skill matches, respond with "none".\n'
            'Respond with ONLY the skill name or "none", nothing else.'
        )

        try:
            result = await self._chat_model.ainvoke(
                [{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (result.content or "").strip().strip('"').strip("'")
            if text in skills:
                return text
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
            logger.warning("[SkillDetector] Detection failed: %s", exc)

        return None


class _NoDecision:
    """Sentinel: the decision path produced no usable answer."""


_NO_DECISION = _NoDecision()
