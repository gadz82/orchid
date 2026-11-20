"""Graph wiring tests for pipeline agent executors.

The pipeline agent step depends on the executor the graph builder would use for
the same agent. These tests assert that ``build_graph`` exposes one executor per
top-level leaf agent, with the global + per-agent guardrail chains merged, the
model router attached and inline mini-agents enabled, while parent agents are
left to their subgraph.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.graph import build_graph
from orchid_ai.runtime import OrchidRuntime

MODEL = "test/model"


def _agent_entry(*, decisions: dict | None = None, guardrails: dict | None = None, mini: bool = False) -> dict:
    entry: dict = {
        "description": "test agent",
        "prompt": "you are a test agent",
        "llm": {"model": MODEL},
        "rag": {"namespace": "kb", "retrieval": {"strategy": "simple"}},
    }
    if decisions is not None:
        entry["decisions"] = {"points": {"model_routing": decisions}}
    if guardrails is not None:
        entry["guardrails"] = guardrails
    if mini:
        entry["mini_agent"] = {"enabled": True, "max_count": 2}
    return entry


def _config() -> OrchidAgentsConfig:
    return OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "guardrails": {
                "input": [
                    {"type": "max_length", "fail_action": "block", "config": {"max_characters": 10000}},
                ],
                "output": [
                    {"type": "max_length", "fail_action": "block", "config": {"max_characters": 10000}},
                ],
            },
            "agents": {
                "a": _agent_entry(
                    decisions={"model": "stub"},
                    guardrails={
                        "input": [{"type": "prompt_injection", "fail_action": "block"}],
                        "output": [{"type": "prompt_injection", "fail_action": "block"}],
                    },
                    mini=True,
                ),
                "team": {
                    **_agent_entry(),
                    "children": {"kid": _agent_entry()},
                },
            },
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )


def _runtime() -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    provider = StubDecisionModel(answers={"model": {"choice": "test/alt", "confidence": 0.95}})
    return OrchidRuntime(default_model=MODEL, chat_model=chat, decision_models={"stub": provider})


def test_build_graph_exposes_leaf_executors_with_guardrails_and_routing() -> None:
    agents: dict = {}
    executors: dict = {}

    build_graph(config=_config(), runtime=_runtime(), agents_out=agents, executors_out=executors)

    assert set(executors) == {"a"}
    executor = executors["a"]
    assert executor.agent is agents["a"]
    assert executor._inline_mini_agents is True
    # Pipeline jobs are independent activations: no shared chat turn budget.
    assert executor._turn_budgets is None
    assert [rail.name for rail in executor._input_guardrails.guardrails] == [
        "max_length",
        "prompt_injection",
    ]
    assert [rail.name for rail in executor._output_guardrails.guardrails] == [
        "prompt_injection",
        "max_length",
    ]
    assert executor._model_router is not None
    assert executor._model_router.point == "model_routing"


def test_build_graph_omits_executors_when_not_requested() -> None:
    agents: dict = {}

    build_graph(config=_config(), runtime=_runtime(), agents_out=agents)

    assert set(agents) == {"a"}
