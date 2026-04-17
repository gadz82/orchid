"""Skill detection — matches user queries to agent-level skills via LLM."""

from __future__ import annotations

import logging
from typing import Any

from ..config.schema import AgentSkillConfig

logger = logging.getLogger(__name__)


class SkillDetector:
    """Detects whether a user query matches an agent-level skill."""

    def __init__(self, chat_model: Any):
        self._chat_model = chat_model

    async def detect(self, query: str, skills: dict[str, AgentSkillConfig]) -> str | None:
        """Return the matching skill name, or None."""
        if not skills:
            return None

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
