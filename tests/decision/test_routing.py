"""Tests for the supervisor routing strategy seam."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.decision import OrchidDecisionError
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.routing import DecisionRoutingStrategy, build_routing_strategy
from orchid_ai.graph.supervisor import OrchidRoutingDecision, _route
from orchid_ai.runtime import OrchidRuntime

AGENTS = {"billing": "Handles invoices and refunds.", "technical": "Handles bugs and outages."}


def _state(text: str = "I was charged twice") -> dict[str, Any]:
    return {"messages": [HumanMessage(content=text)], "chat_id": "c-1"}


class FakeStructured:
    def __init__(self, decision: OrchidRoutingDecision) -> None:
        self.decision = decision
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return self.decision


class FakeChat:
    def __init__(self, decision: OrchidRoutingDecision) -> None:
        self.structured = FakeStructured(decision)

    def with_structured_output(self, schema):
        return self.structured


def _provider(answers: dict[str, dict[str, Any]]) -> StubDecisionModel:
    return StubDecisionModel(answers=answers)


async def test_decision_strategy_selects_agents_and_execution() -> None:
    strategy = DecisionRoutingStrategy(
        provider=_provider(
            {
                "agent:billing": {"probability": 0.95},
                "agent:technical": {"probability": 0.1},
                "execution": {"choice": "parallel"},
            }
        ),
        confidence_threshold=0.8,
        provider_name="fast",
    )
    decision = await strategy.route(state=_state(), agent_descriptions=AGENTS)
    assert decision is not None
    assert decision.agents == ["billing"]
    assert decision.execution == "parallel"
    assert strategy.metadata == {"provider": "fast", "threshold": 0.8}


async def test_decision_strategy_selects_multiple_agents() -> None:
    strategy = DecisionRoutingStrategy(
        provider=_provider(
            {
                "agent:billing": {"probability": 0.95},
                "agent:technical": {"probability": 0.9},
                "execution": {"choice": "sequential"},
            }
        ),
        confidence_threshold=0.8,
    )
    decision = await strategy.route(state=_state(), agent_descriptions=AGENTS)
    assert decision is not None
    assert decision.agents == ["billing", "technical"]
    assert decision.execution == "sequential"


async def test_decision_strategy_returns_none_below_threshold() -> None:
    strategy = DecisionRoutingStrategy(provider=StubDecisionModel(mode="low_confidence"), confidence_threshold=0.9)
    assert await strategy.route(state=_state(), agent_descriptions=AGENTS) is None


async def test_decision_strategy_selects_a_skill() -> None:
    skills = {"refund_request": type("Skill", (), {"description": "Refund workflow"})()}
    strategy = DecisionRoutingStrategy(
        provider=_provider({"skill": {"choice": "refund_request"}}),
        confidence_threshold=0.8,
    )
    decision = await strategy.route(state=_state(), agent_descriptions=AGENTS, orchestrator_skills=skills)
    assert decision is not None
    assert decision.execution == "skill"
    assert decision.skill == "refund_request"
    assert decision.agents == []


async def test_decision_strategy_raises_on_provider_failure() -> None:
    strategy = DecisionRoutingStrategy(provider=StubDecisionModel(mode="fail"))
    with pytest.raises(OrchidDecisionError):
        await strategy.route(state=_state(), agent_descriptions=AGENTS)


def test_build_routing_strategy_returns_none_without_configuration() -> None:
    config = OrchidAgentsConfig.model_validate({"agents": {"a": {"description": "d", "prompt": "p"}}})
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    assert build_routing_strategy(config, runtime) is None


def test_build_routing_strategy_returns_none_when_provider_is_missing() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {"a": {"description": "d", "prompt": "p"}},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {"routing": {"model": "stub"}},
        }
    )
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    assert build_routing_strategy(config, runtime) is None


def test_build_routing_strategy_uses_the_runtime_provider() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {"a": {"description": "d", "prompt": "p"}},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {"routing": {"model": "stub", "confidence_threshold": 0.7}},
        }
    )
    runtime = OrchidRuntime(default_model="ollama/llama3.2", decision_models={"stub": StubDecisionModel()})
    strategy = build_routing_strategy(config, runtime)
    assert isinstance(strategy, DecisionRoutingStrategy)
    assert strategy.metadata["threshold"] == 0.7


async def test_route_uses_the_decision_strategy_and_skips_the_llm() -> None:
    strategy = DecisionRoutingStrategy(
        provider=_provider(
            {
                "agent:billing": {"probability": 0.95},
                "agent:technical": {"probability": 0.1},
                "execution": {"choice": "parallel"},
            }
        ),
        confidence_threshold=0.8,
        provider_name="fast",
    )
    chat = FakeChat(OrchidRoutingDecision(reasoning="llm", agents=["technical"]))
    result = await _route(
        _state(),
        "ollama/llama3.2",
        AGENTS,
        chat_model=chat,
        routing=strategy,
    )
    assert result["active_agents"] == ["billing"]
    assert result["routing_metadata"]["path"] == "decision"
    assert result["routing_metadata"]["provider"] == "fast"
    assert chat.structured.calls == 0


async def test_route_falls_back_to_the_llm_on_low_confidence() -> None:
    strategy = DecisionRoutingStrategy(provider=StubDecisionModel(mode="low_confidence"), confidence_threshold=0.9)
    chat = FakeChat(OrchidRoutingDecision(reasoning="llm", agents=["technical"]))
    result = await _route(
        _state(),
        "ollama/llama3.2",
        AGENTS,
        chat_model=chat,
        routing=strategy,
    )
    assert result["active_agents"] == ["technical"]
    assert result["routing_metadata"]["path"] == "llm"
    assert chat.structured.calls == 1


async def test_route_falls_back_to_the_llm_when_the_provider_fails() -> None:
    strategy = DecisionRoutingStrategy(provider=StubDecisionModel(mode="fail"))
    chat = FakeChat(OrchidRoutingDecision(reasoning="llm", agents=["billing"]))
    result = await _route(
        _state(),
        "ollama/llama3.2",
        AGENTS,
        chat_model=chat,
        routing=strategy,
    )
    assert result["active_agents"] == ["billing"]
    assert result["routing_metadata"]["path"] == "llm"
