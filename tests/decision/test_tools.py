"""Tests for the decision-first tool shortlist."""

from __future__ import annotations

from typing import Any

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import OrchidAgentConfig
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.rag.backends.null import NullVectorReader


def _agent(provider: Any | None, *, max_candidates: int = 32) -> GenericAgent:
    return GenericAgent(
        config=OrchidAgentConfig(description="test", prompt="test", name="a"),
        reader=NullVectorReader(),
        tool_decision_model=provider,
        tool_max_candidates=max_candidates,
    )


def _tools(*names: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name, "description": f"{name} tool"}} for name in names]


async def test_confident_shortlist_returns_selected_tools() -> None:
    provider = StubDecisionModel(answers={"tool:search": {"probability": 0.95}, "tool:delete": {"probability": 0.1}})
    agent = _agent(provider)
    assert await agent._shortlist_tools("find docs", _tools("search", "delete")) == ["search"]


async def test_low_confidence_binds_all_tools() -> None:
    agent = _agent(StubDecisionModel(mode="low_confidence"))
    assert await agent._shortlist_tools("find docs", _tools("search", "delete")) is None


async def test_provider_failure_binds_all_tools() -> None:
    agent = _agent(StubDecisionModel(mode="fail"))
    assert await agent._shortlist_tools("find docs", _tools("search", "delete")) is None


async def test_no_provider_keeps_the_full_toolset() -> None:
    agent = _agent(None)
    assert await agent._shortlist_tools("find docs", _tools("search", "delete")) is None


async def test_candidate_cap_binds_all_tools() -> None:
    agent = _agent(StubDecisionModel(), max_candidates=2)
    assert await agent._shortlist_tools("find docs", _tools("a", "b", "c")) is None


async def test_empty_tool_list_returns_none() -> None:
    agent = _agent(StubDecisionModel())
    assert await agent._shortlist_tools("find docs", []) is None


async def test_unknown_tool_names_are_ignored() -> None:
    class GhostProvider:
        async def decide(self, state, questions, *, timeout=None):
            from orchid_ai.core.decision import OrchidDecisionAnswer, OrchidDecisionResult

            return OrchidDecisionResult(
                answers={
                    "tool:search": OrchidDecisionAnswer(name="tool:search", kind="boolean", value=True, confidence=0.9),
                    "tool:ghost": OrchidDecisionAnswer(name="tool:ghost", kind="boolean", value=True, confidence=0.99),
                },
                provider="test",
                model="test",
            )

        async def aclose(self):
            return None

    agent = _agent(GhostProvider())
    assert await agent._shortlist_tools("find docs", _tools("search")) == ["search"]
