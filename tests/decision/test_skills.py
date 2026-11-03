"""Tests for decision-first agent-level skill detection."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from orchid_ai.agents.skill_detector import SkillDetector
from orchid_ai.decision.stub import StubDecisionModel

SKILLS = {
    "refund_request": type("Skill", (), {"description": "Handle a refund request"})(),
    "outage_report": type("Skill", (), {"description": "Report an outage"})(),
}


class FakeChat:
    def __init__(self, answer: str = "refund_request") -> None:
        self.answer = answer
        self.calls = 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self.answer)


async def test_decision_selects_a_skill_without_the_llm() -> None:
    chat = FakeChat()
    detector = SkillDetector(
        chat,
        decision_model=StubDecisionModel(answers={"skill": {"choice": "refund_request"}}),
        confidence_threshold=0.8,
    )
    assert await detector.detect("please refund me", SKILLS) == "refund_request"
    assert chat.calls == 0


async def test_decision_none_skips_the_llm() -> None:
    chat = FakeChat()
    detector = SkillDetector(
        chat,
        decision_model=StubDecisionModel(answers={"skill": {"choice": "__none__"}}),
    )
    assert await detector.detect("hello", SKILLS) is None
    assert chat.calls == 0


async def test_low_confidence_falls_back_to_the_llm() -> None:
    chat = FakeChat("outage_report")
    detector = SkillDetector(
        chat,
        decision_model=StubDecisionModel(mode="low_confidence"),
        confidence_threshold=0.9,
    )
    assert await detector.detect("the site is down", SKILLS) == "outage_report"
    assert chat.calls == 1


async def test_provider_failure_falls_back_to_the_llm() -> None:
    chat = FakeChat("refund_request")
    detector = SkillDetector(
        chat,
        decision_model=StubDecisionModel(mode="fail"),
    )
    assert await detector.detect("please refund me", SKILLS) == "refund_request"
    assert chat.calls == 1


async def test_no_decision_model_keeps_the_llm_path() -> None:
    chat = FakeChat("refund_request")
    detector = SkillDetector(chat)
    assert await detector.detect("please refund me", SKILLS) == "refund_request"
    assert chat.calls == 1


async def test_decision_works_without_a_chat_model() -> None:
    detector = SkillDetector(
        None,
        decision_model=StubDecisionModel(answers={"skill": {"choice": "outage_report"}}),
    )
    assert await detector.detect("the site is down", SKILLS) == "outage_report"


async def test_low_confidence_without_a_chat_model_returns_none() -> None:
    detector = SkillDetector(None, decision_model=StubDecisionModel(mode="low_confidence"), confidence_threshold=0.9)
    assert await detector.detect("hello", SKILLS) is None


async def test_no_skills_returns_none_without_calling_anything() -> None:
    chat = FakeChat()
    detector = SkillDetector(chat, decision_model=StubDecisionModel())
    assert await detector.detect("hello", {}) is None
    assert chat.calls == 0


class HandRolledProvider:
    def __init__(self, answer) -> None:
        self.answer = answer

    async def decide(self, state, questions, *, timeout=None):
        from orchid_ai.core.decision import OrchidDecisionResult

        return OrchidDecisionResult(answers={"skill": self.answer}, provider="test", model="test")

    async def aclose(self):
        return None


async def test_unknown_skill_falls_back_to_the_llm() -> None:
    from orchid_ai.core.decision import OrchidDecisionAnswer

    chat = FakeChat("refund_request")
    provider = HandRolledProvider(OrchidDecisionAnswer(name="skill", kind="choice", value="ghost", confidence=0.99))
    detector = SkillDetector(chat, decision_model=provider)
    assert await detector.detect("hello", SKILLS) == "refund_request"
    assert chat.calls == 1


async def test_timeout_falls_back_to_the_llm() -> None:
    from orchid_ai.core.decision import OrchidDecisionError

    class TimeoutProvider:
        async def decide(self, state, questions, *, timeout=None):
            raise OrchidDecisionError("timeout", code="timeout", retryable=True)

        async def aclose(self):
            return None

    chat = FakeChat("outage_report")
    detector = SkillDetector(chat, decision_model=TimeoutProvider())
    assert await detector.detect("the site is down", SKILLS) == "outage_report"
    assert chat.calls == 1


async def test_reserved_sentinel_skill_falls_back() -> None:
    chat = FakeChat("refund_request")
    detector = SkillDetector(
        chat,
        decision_model=StubDecisionModel(answers={"skill": {"choice": "__none__"}}),
    )
    skills = {**SKILLS, "__none__": type("Skill", (), {"description": "reserved"})()}
    assert await detector.detect("hello", skills) == "refund_request"
    assert chat.calls == 1
