"""Tests for the deterministic stub decision provider."""

from __future__ import annotations

import pytest

from orchid_ai.core.decision import OrchidDecisionError, OrchidDecisionQuestion, OrchidDecisionState
from orchid_ai.decision.stub import StubDecisionModel


def _questions() -> list[OrchidDecisionQuestion]:
    return [
        OrchidDecisionQuestion(name="relevant", kind="boolean"),
        OrchidDecisionQuestion(name="route", kind="choice", options=("billing", "technical")),
        OrchidDecisionQuestion(name="urgency", kind="score", rubric=("0", "1", "2")),
    ]


async def test_defaults_are_deterministic() -> None:
    model = StubDecisionModel()
    result = await model.decide(OrchidDecisionState(text="x"), _questions())
    assert result.answers["relevant"].value is True
    assert result.answers["route"].value == "billing"
    assert result.answers["urgency"].value == 0.0

    again = await model.decide(OrchidDecisionState(text="x"), _questions())
    assert result.answers == again.answers


async def test_custom_answers() -> None:
    model = StubDecisionModel(
        answers={
            "relevant": {"value": False},
            "route": {"choice": "technical"},
            "urgency": {"score": 2.0},
        }
    )
    result = await model.decide(OrchidDecisionState(text="x"), _questions())
    assert result.answers["relevant"].value is False
    assert result.answers["route"].value == "technical"
    assert result.answers["urgency"].value == 2.0


async def test_contextual_answer_wins() -> None:
    model = StubDecisionModel(answers={"charged twice:route": {"choice": "technical"}})
    result = await model.decide(OrchidDecisionState(text="charged twice"), _questions())
    assert result.answers["route"].value == "technical"


async def test_low_confidence_mode() -> None:
    model = StubDecisionModel(mode="low_confidence")
    result = await model.decide(OrchidDecisionState(text="x"), _questions())
    assert result.answers["relevant"].confidence == pytest.approx(0.55)
    assert result.answers["route"].confidence == pytest.approx(0.55)


async def test_fail_mode() -> None:
    model = StubDecisionModel(mode="fail")
    with pytest.raises(OrchidDecisionError) as excinfo:
        await model.decide(OrchidDecisionState(text="x"), _questions())
    assert excinfo.value.code == "provider_unavailable"


async def test_refusal_spec() -> None:
    model = StubDecisionModel(answers={"route": {"refusal": "not applicable"}})
    result = await model.decide(OrchidDecisionState(text="x"), _questions())
    assert result.answers["route"].refusal == "not applicable"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        StubDecisionModel(mode="sometimes")  # type: ignore[arg-type]
