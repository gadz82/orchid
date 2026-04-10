"""Skill detection — matches user queries to agent-level skills via LLM."""
from __future__ import annotations

import logging

from ..core.llm_provider import LLMProvider
from ..config.schema import AgentSkillConfig

logger = logging.getLogger(__name__)


class SkillDetector:
    """Detects whether a user query matches an agent-level skill."""

    def __init__(self, llm_service: LLMProvider, model: str):
        self._llm_service = llm_service
        self._model = model

    async def detect(self, query: str, skills: dict[str, AgentSkillConfig]) -> str | None:
        """Return the matching skill name, or None."""
        if not skills:
            return None

        skill_descriptions = "\n".join(
            f'- "{name}": {skill.description}' for name, skill in skills.items()
        )

        prompt = (
            f"User query: {query}\n\n"
            f"Available skills for this agent:\n{skill_descriptions}\n\n"
            "If the user's query closely matches one of these skills, respond with "
            'ONLY the skill name (e.g. "course_completion_summary").\n'
            'If no skill matches, respond with "none".\n'
            'Respond with ONLY the skill name or "none", nothing else.'
        )

        try:
            result = await self._llm_service.complete(
                self._model,
                [{"role": "user", "content": prompt}],
                temperature=0,
            )
            result = result.strip().strip('"').strip("'")
            if result in skills:
                return result
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
            logger.warning("[SkillDetector] Detection failed: %s", exc)

        return None
