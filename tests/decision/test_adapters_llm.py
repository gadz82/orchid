"""Tests for the LLM-backed decision provider."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orchid_ai.core.decision import OrchidDecisionError, OrchidDecisionQuestion, OrchidDecisionState
from orchid_ai.decision.llm import LLMAnswerSchema, LLMDecisionModel, LLMResultSchema


class FakeStructured:
    def __init__(self, result: Any = None, *, error: Exception | None = None, delay: float = 0.0) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.messages = messages
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


class FakeChatModel:
    def __init__(self, result: Any = None, *, error: Exception | None = None, delay: float = 0.0) -> None:
        self.structured = FakeStructured(result, error=error, delay=delay)
        self.schema: Any = None

    def with_structured_output(self, schema: Any) -> FakeStructured:
        self.schema = schema
        return self.structured


def _model(result: Any = None, **kwargs) -> tuple[LLMDecisionModel, FakeChatModel]:
    chat = FakeChatModel(result, **kwargs)
    return LLMDecisionModel(chat_model=chat, model="chat-model", provider="local", timeout=1.0), chat


def _questions() -> list[OrchidDecisionQuestion]:
    return [
        OrchidDecisionQuestion(name="relevant", kind="boolean"),
        OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
        OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2")),
    ]


def _result() -> LLMResultSchema:
    return LLMResultSchema(
        answers=[
            LLMAnswerSchema(name="relevant", probability=0.91),
            LLMAnswerSchema(name="route", choice="billing", confidence=0.87),
            LLMAnswerSchema(name="urgency", score=1.2, confidence=0.8),
        ]
    )


async def test_happy_path_all_three_kinds() -> None:
    model, chat = _model(_result())
    result = await model.decide(OrchidDecisionState(text="charged twice"), _questions())
    assert result.answers["relevant"].value is True
    assert result.answers["route"].value == "billing"
    assert result.answers["urgency"].value == 1.2
    assert chat.schema is LLMResultSchema


async def test_boolean_false() -> None:
    model, _ = _model(LLMResultSchema(answers=[LLMAnswerSchema(name="q", probability=0.2)]))
    result = await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is False
    assert result.answers["q"].confidence == pytest.approx(0.8)


async def test_refusal_is_non_fatal() -> None:
    model, _ = _model(
        LLMResultSchema(
            answers=[
                LLMAnswerSchema(name="relevant", probability=0.9),
                LLMAnswerSchema(name="route", refusal="not enough context"),
            ]
        )
    )
    result = await model.decide(OrchidDecisionState(text="x"), _questions()[:2])
    assert result.answers["relevant"].value is True
    assert result.answers["route"].refusal == "not enough context"


async def test_missing_required_answer_is_invalid() -> None:
    model, _ = _model(LLMResultSchema(answers=[]))
    with pytest.raises(OrchidDecisionError) as excinfo:
        await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "invalid_response"


async def test_optional_missing_answer_is_omitted() -> None:
    model, _ = _model(LLMResultSchema(answers=[]))
    result = await model.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean", required=False)],
    )
    assert result.answers == {}


async def test_unknown_and_duplicate_names_are_invalid() -> None:
    model, _ = _model(LLMResultSchema(answers=[LLMAnswerSchema(name="extra", probability=0.9)]))
    with pytest.raises(OrchidDecisionError):
        await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])

    model, _ = _model(
        LLMResultSchema(
            answers=[LLMAnswerSchema(name="q", probability=0.9), LLMAnswerSchema(name="q", probability=0.1)]
        )
    )
    with pytest.raises(OrchidDecisionError):
        await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])


async def test_invalid_confidence_is_invalid() -> None:
    model, _ = _model(LLMResultSchema(answers=[LLMAnswerSchema(name="route", choice="billing", confidence=1.5)]))
    with pytest.raises(OrchidDecisionError) as excinfo:
        await model.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical"))],
        )
    assert excinfo.value.code == "invalid_response"


async def test_timeout_maps_to_timeout_code() -> None:
    model, _ = _model(_result(), delay=0.2)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await model.decide(OrchidDecisionState(text="x"), _questions(), timeout=0.05)
    assert excinfo.value.code == "timeout"


async def test_chat_model_failure_maps_to_provider_unavailable() -> None:
    model, _ = _model(error=RuntimeError("boom"))
    with pytest.raises(OrchidDecisionError) as excinfo:
        await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert excinfo.value.code == "provider_unavailable"


async def test_prompt_carries_state_and_questions() -> None:
    model, chat = _model(_result())
    await model.decide(OrchidDecisionState(text="charged twice", payload={"order": "A-1"}), _questions())
    content = "\n".join(str(message.content) for message in chat.structured.messages)
    assert "charged twice" in content
    assert "A-1" in content
    assert "route" in content
    assert "billing" in content


async def test_result_may_be_a_dict() -> None:
    model, _ = _model({"answers": [{"name": "q", "probability": 0.9}]})
    result = await model.decide(OrchidDecisionState(text="x"), [OrchidDecisionQuestion(name="q", kind="boolean")])
    assert result.answers["q"].value is True
