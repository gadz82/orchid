from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.skill_detector import SkillDetector
from orchid_ai.config.schema import OrchidAgentSkillConfig


def _make_skill(name: str, description: str) -> OrchidAgentSkillConfig:
    return OrchidAgentSkillConfig(name=name, description=description, steps=[])


class TestSkillDetector:
    @pytest.fixture
    def detector(self):
        model = MagicMock()
        model.ainvoke = AsyncMock()
        return SkillDetector(model)

    @pytest.mark.asyncio
    async def test_empty_skills_returns_none(self, detector):
        result = await detector.detect("any query", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_matching_skill_returns_skill_name(self, detector):
        detector._chat_model.ainvoke.return_value = MagicMock(content="greeting")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user warmly"),
            "farewell": _make_skill("farewell", "Say goodbye"),
        }
        result = await detector.detect("hello", skills)
        assert result == "greeting"

    @pytest.mark.asyncio
    async def test_skill_name_with_quotes_is_stripped(self, detector):
        detector._chat_model.ainvoke.return_value = MagicMock(content='"greeting"')
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result == "greeting"

    @pytest.mark.asyncio
    async def test_skill_name_with_single_quotes_stripped(self, detector):
        detector._chat_model.ainvoke.return_value = MagicMock(content="'greeting'")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result == "greeting"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, detector):
        detector._chat_model.ainvoke.return_value = MagicMock(content="none")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("unrelated query", skills)
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self, detector):
        detector._chat_model.ainvoke.side_effect = ConnectionError("API down")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_error_returns_none(self, detector):
        detector._chat_model.ainvoke.side_effect = TimeoutError("timeout")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result is None

    @pytest.mark.asyncio
    async def test_value_error_returns_none(self, detector):
        detector._chat_model.ainvoke.side_effect = ValueError("bad value")
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result is None

    @pytest.mark.asyncio
    async def test_none_content_returns_none(self, detector):
        detector._chat_model.ainvoke.return_value = MagicMock(content=None)
        skills = {
            "greeting": _make_skill("greeting", "Greet the user"),
        }
        result = await detector.detect("hello", skills)
        assert result is None
