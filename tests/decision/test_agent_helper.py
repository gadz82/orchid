"""Tests for the ``OrchidAgent.decide`` helper and its config wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.decision import OrchidDecisionQuestion, OrchidDecisionState
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.rag.backends.null import NullVectorReader


class _Agent(OrchidAgent):
    @property
    def name(self) -> str:
        return "helper"

    @property
    def description(self) -> str:
        return "test agent"

    @property
    def rag_namespace(self) -> str:
        return "helper"

    async def run(self, state):
        return {}


async def test_decide_uses_the_injected_provider() -> None:
    provider = StubDecisionModel(answers={"q": {"probability": 0.2}})
    agent = _Agent(reader=NullVectorReader(), decision_model=provider)
    result = await agent.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean")],
    )
    assert result is not None
    assert result.answers["q"].value is False


async def test_decide_returns_none_without_a_provider() -> None:
    agent = _Agent(reader=NullVectorReader())
    result = await agent.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean")],
    )
    assert result is None


async def test_decide_accepts_an_explicit_provider() -> None:
    agent = _Agent(reader=NullVectorReader())
    result = await agent.decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean")],
        provider=StubDecisionModel(answers={"q": {"probability": 0.9}}),
    )
    assert result is not None
    assert result.answers["q"].value is True


def test_unknown_agent_decision_model_fails_validation() -> None:
    with pytest.raises(ValidationError, match="unknown decision provider"):
        OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "a": {"description": "d", "prompt": "p", "decision_model": "missing"},
                }
            }
        )


def test_agent_decision_model_is_built_even_without_a_point_binding() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {
                "a": {"description": "d", "prompt": "p", "decision_model": "stub"},
            },
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    providers = build_decision_models(config)
    assert isinstance(providers["stub"], StubDecisionModel)
