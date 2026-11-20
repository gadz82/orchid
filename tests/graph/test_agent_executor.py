"""Parity tests for the shared :class:`OrchidAgentExecutor`.

Every test runs the same fake agent through the chat wrapper
(``_create_agent_node``) and through the executor directly, then compares the
observable outcome: returned state update, run-context values, guardrail
decisions, routed model selection, mini-agent aggregation and error
translation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp

from orchid_ai.config.schema import OrchidAgentConfig, OrchidMiniAgentConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailResult,
)
from orchid_ai.core.run_config import auth_from_config, with_auth
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.agent_executor import OrchidAgentExecutor
from orchid_ai.graph.decision_bindings import DecisionBinding
from orchid_ai.graph.graph import _create_agent_node
from orchid_ai.graph.model_routing import (
    ModelRoutingBudget,
    ModelRoutingSelector,
    TurnBudgetRegistry,
    build_routed_chat_model,
)

# ── Fakes ─────────────────────────────────────────────────────


class _RecordingAgent(OrchidAgent):
    """Agent that records the run context it observes and returns a fixed update."""

    def __init__(
        self,
        name: str = "a",
        *,
        chat_model: Any | None = None,
        result: dict[str, Any] | None = None,
        raises: BaseException | None = None,
        probe: Any | None = None,
    ) -> None:
        super().__init__(reader=None, chat_model=chat_model)  # type: ignore[arg-type]
        self._name = name
        self._result = result or {"messages": [AIMessage(content="answer")]}
        self._raises = raises
        self._probe = probe
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "recording agent"

    async def run(self, state):
        context = self.current_run_context()
        self.calls.append(
            {
                "query": self.extract_user_query(state),
                "auth": self._current_auth,
                "chat_id": self._current_chat_id,
                "message_id": self._current_message_id,
                "correlation_id": self._current_correlation_id,
                "persist_memory": context.persist_memory if context is not None else None,
                "probe": self._probe() if self._probe is not None else None,
                "state": state,
            }
        )
        if self._raises is not None:
            raise self._raises
        return dict(self._result)


class _StaticRail(OrchidGuardrail):
    def __init__(self, name: str, result: OrchidGuardrailResult) -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return self._result


def _blocking_rail(message: str = "blocked") -> OrchidGuardrail:
    return _StaticRail(
        "block",
        OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name="block",
            message=message,
        ),
    )


def _redacting_rail(replacement: str = "[REDACTED]") -> OrchidGuardrail:
    return _StaticRail(
        "redact",
        OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.REDACT,
            guardrail_name="redact",
            redacted_content=replacement,
        ),
    )


class _RecordingRouter:
    """Minimal stand-in for ``ModelRoutingSelector`` exposing ``bound``."""

    point = "model_routing"
    default_model = "default-model"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.budgets: list[ModelRoutingBudget | None] = []
        self.active = False

    @asynccontextmanager
    async def bound(self, query: str, *, budget: ModelRoutingBudget | None = None):
        self.queries.append(query)
        self.budgets.append(budget)
        self.active = True
        try:
            yield "routed-model"
        finally:
            self.active = False


class _StubChatModel:
    """Only the aggregator's ``ainvoke`` is needed by the mini-agent parity test."""

    def __init__(self, content: str = "aggregated answer") -> None:
        self.content = content
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.content)


def _state(*, chat_id: str = "c-1", query: str = "the query") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query, id="m-1")],
        "chat_id": chat_id,
        "active_agents": ["a"],
    }


def _message_view(messages: list[Any]) -> list[tuple[str, Any]]:
    return [(type(message).__name__, getattr(message, "content", None)) for message in messages]


def _mini_config() -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name="a",
        description="mini agent",
        prompt="prompt",
        mini_agent=OrchidMiniAgentConfig(enabled=True, max_count=3),
    )


def _decomposition(count: int = 2) -> dict[str, Any]:
    sub_tasks = [
        {
            "id": f"mini_{index}",
            "description": f"task {index}",
            "instruction": f"do {index}",
            "allowed_tools": [f"tool_{index}"],
            "rationale": "independent",
            "resolved_tool_subset": [f"tool_{index}"],
        }
        for index in range(count)
    ]
    return {
        "mini_agent_decisions": {
            "a": {"should_fork": True, "sub_tasks": sub_tasks, "reasoning": "independent"},
        },
        "messages": [],
    }


# ── Run context parity ────────────────────────────────────────


async def test_wrapper_and_executor_bind_the_same_run_context() -> None:
    auth = OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1")
    state = _state()

    wrapper_agent = _RecordingAgent()
    wrapper_agent._current_correlation_id = "corr-1"
    wrapper_node = _create_agent_node(wrapper_agent)
    wrapper_result = await wrapper_node(state, with_auth(auth))

    executor_agent = _RecordingAgent()
    executor = OrchidAgentExecutor(executor_agent)
    executor_result = await executor.run(
        state,
        auth=auth,
        chat_id=state["chat_id"],
        message_id="m-1",
        correlation_id="corr-1",
        persist_memory=True,
    )

    assert wrapper_agent.calls[0] == executor_agent.calls[0]
    assert wrapper_agent.calls[0]["auth"] is auth
    assert wrapper_agent.calls[0]["chat_id"] == "c-1"
    assert wrapper_agent.calls[0]["message_id"] == "m-1"
    assert wrapper_agent.calls[0]["correlation_id"] == "corr-1"
    assert wrapper_agent.calls[0]["persist_memory"] is True

    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])
    assert wrapper_result["active_agents"] == executor_result["active_agents"] == []


async def test_persist_memory_false_reaches_the_run_context() -> None:
    agent = _RecordingAgent()
    executor = OrchidAgentExecutor(agent)

    await executor.run(_state(), auth=None, chat_id="c-1", message_id="m-1", persist_memory=False)

    assert agent.calls[0]["persist_memory"] is False


# ── Guardrail parity ─────────────────────────────────────────


async def test_input_block_parity() -> None:
    chain = OrchidGuardrailChain([_blocking_rail("blocked input")])
    state = _state()

    wrapper_agent = _RecordingAgent()
    wrapper_result = await _create_agent_node(wrapper_agent, input_guardrails=chain)(state)

    executor_agent = _RecordingAgent()
    executor = OrchidAgentExecutor(executor_agent, input_guardrails=chain)
    executor_result = await executor.run(state, auth=None)

    assert wrapper_agent.calls == executor_agent.calls == []
    assert _message_view(wrapper_result["messages"]) == [("AIMessage", "[A Agent] blocked input")]
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])
    assert wrapper_result["active_agents"] == executor_result["active_agents"] == []


@pytest.mark.asyncio
async def test_input_redaction_parity() -> None:
    raw_query = "mail user@example.com"
    expected = "[REDACTED]"
    chain = OrchidGuardrailChain([_redacting_rail()])
    state = {
        "messages": [HumanMessage(content=raw_query, id="m-1")],
        "chat_id": "c-1",
        "active_agents": ["a"],
    }

    wrapper_agent = _RecordingAgent()
    wrapper_result = await _create_agent_node(wrapper_agent, input_guardrails=chain)(state)

    executor_agent = _RecordingAgent()
    executor = OrchidAgentExecutor(executor_agent, input_guardrails=chain)
    executor_result = await executor.run(state, auth=None)

    assert wrapper_agent.calls[0]["query"] == expected
    assert executor_agent.calls[0]["query"] == expected
    # The replacement user turn is part of the returned update on both paths.
    assert _message_view(wrapper_result["messages"])[0] == ("HumanMessage", expected)
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])


async def test_output_block_parity() -> None:
    chain = OrchidGuardrailChain([_blocking_rail("blocked output")])
    state = _state()

    wrapper_agent = _RecordingAgent()
    wrapper_result = await _create_agent_node(wrapper_agent, output_guardrails=chain)(state)

    executor_agent = _RecordingAgent()
    executor = OrchidAgentExecutor(executor_agent, output_guardrails=chain)
    executor_result = await executor.run(state, auth=None)

    assert _message_view(wrapper_result["messages"]) == [("AIMessage", "[A Agent] blocked output")]
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])


# ── Model routing parity ─────────────────────────────────────


async def test_model_routing_selects_once_on_both_paths() -> None:
    state = _state()

    wrapper_router = _RecordingRouter()
    wrapper_agent = _RecordingAgent(probe=lambda: wrapper_router.active)
    wrapper_node = _create_agent_node(
        wrapper_agent,
        model_router=wrapper_router,
        turn_budgets=TurnBudgetRegistry(max_calls_per_turn=2),
    )
    wrapper_result = await wrapper_node(state)

    executor_router = _RecordingRouter()
    executor_agent = _RecordingAgent(probe=lambda: executor_router.active)
    executor = OrchidAgentExecutor(
        executor_agent,
        model_router=executor_router,
        turn_budgets=TurnBudgetRegistry(max_calls_per_turn=2),
    )
    executor_result = await executor.run(state, auth=None)

    assert wrapper_router.queries == executor_router.queries == ["the query"]
    assert wrapper_agent.calls[0]["probe"] is True
    assert executor_agent.calls[0]["probe"] is True
    assert isinstance(wrapper_router.budgets[0], ModelRoutingBudget)
    assert isinstance(executor_router.budgets[0], ModelRoutingBudget)
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])


async def test_routing_is_not_consulted_when_input_is_blocked() -> None:
    router = _RecordingRouter()
    chain = OrchidGuardrailChain([_blocking_rail()])
    executor = OrchidAgentExecutor(_RecordingAgent(), input_guardrails=chain, model_router=router)

    await executor.run(_state(), auth=None)

    assert router.queries == []


async def test_routing_decision_is_reported_through_trace_out() -> None:
    router = _RecordingRouter()
    trace: dict[str, Any] = {}
    executor = OrchidAgentExecutor(_RecordingAgent(), model_router=router)

    await executor.run(_state(), auth=None, trace_out=trace)

    assert trace["model_routing"] == {
        "point": "model_routing",
        "model": "routed-model",
        "default_model": "default-model",
        "routed": True,
    }


async def test_trace_out_without_a_router_records_the_agent_invocation() -> None:
    trace: dict[str, Any] = {}
    executor = OrchidAgentExecutor(_RecordingAgent())

    await executor.run(_state(), auth=None, trace_out=trace)

    assert "model_routing" not in trace
    assert trace["agent_invoked"] is True


async def test_inline_decomposer_rejection_falls_back_to_the_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="[A Agent] I couldn't break this request into sub-tasks")],
            "mcp_context": {"a": {"summary": "rejected"}},
        }

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)

    result = await executor.run(_state(), auth=OrchidAuthContext(access_token="tok"), chat_id="c-1")

    assert len(agent.calls) == 1
    assert [message.content for message in result["messages"]] == ["answer"]
    assert "couldn't break" not in str(result["messages"])


async def test_chat_path_keeps_the_decomposer_rejection_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="[A Agent] I couldn't break this request into sub-tasks")],
            "mcp_context": {"a": {"summary": "rejected"}},
        }

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=False)

    result = await executor.run(_state(), auth=OrchidAuthContext(access_token="tok"), chat_id="c-1")

    assert agent.calls == []
    assert "couldn't break" in result["messages"][-1].content


# ── Error translation parity ─────────────────────────────────


async def test_agent_exception_is_translated_identically() -> None:
    state = _state()

    wrapper_agent = _RecordingAgent(raises=RuntimeError("boom"))
    wrapper_result = await _create_agent_node(wrapper_agent)(state)

    executor_agent = _RecordingAgent(raises=RuntimeError("boom"))
    executor = OrchidAgentExecutor(executor_agent)
    executor_result = await executor.run(state, auth=None)

    expected_message = "[A Agent] I'm temporarily unable to process your request. Please try again in a few moments."
    assert _message_view(wrapper_result["messages"]) == [("AIMessage", expected_message)]
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])
    assert wrapper_result["error"] == executor_result["error"] == "RuntimeError: boom"
    assert wrapper_result["active_agents"] == executor_result["active_agents"] == []
    # The run context is restored even after a crash.
    assert wrapper_agent.current_run_context() is None
    assert executor_agent.current_run_context() is None


async def test_graph_bubble_up_is_not_translated_on_either_path() -> None:
    state = _state()

    wrapper_agent = _RecordingAgent(raises=GraphBubbleUp("interrupt"))
    with pytest.raises(GraphBubbleUp):
        await _create_agent_node(wrapper_agent)(state)

    executor_agent = _RecordingAgent(raises=GraphBubbleUp("interrupt"))
    executor = OrchidAgentExecutor(executor_agent)
    with pytest.raises(GraphBubbleUp):
        await executor.run(state, auth=None)

    assert wrapper_agent.current_run_context() is None
    assert executor_agent.current_run_context() is None


# ── Mini-agent parity ────────────────────────────────────────


def _fake_mini_node_factory(records: list[dict[str, Any]]):
    from orchid_ai.agents.mini_agent_node import _wrap_state_update

    def factory(*, parent_config: OrchidAgentConfig, chat_model: Any, mcp_clients: Any):
        async def node(state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            mini_id = state["_active_mini_id"]
            records.append(
                {
                    "parent": state.get("_active_mini_parent"),
                    "mini_id": mini_id,
                    "sub_task": state.get("_active_mini_subtask"),
                    "tool_subset": state.get("_active_mini_tool_subset"),
                    "auth": auth_from_config(config),
                }
            )
            outcome = {
                "mini_id": mini_id,
                "sub_task_description": state["_active_mini_subtask"]["description"],
                "status": "ok",
                "summary": f"summary-{mini_id}",
                "error": None,
                "duration_ms": 1,
                "tool_results": {f"result-{mini_id}": "value"},
            }
            return _wrap_state_update(parent_config.name, f"{parent_config.name}#{mini_id}", outcome)

        return node

    return factory


async def test_inline_mini_aggregation_matches_the_chat_send_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.agents import mini_agent_node
    from orchid_ai.agents.mini_agent_aggregator import aggregator_node_factory

    decomposition = _decomposition(count=2)

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return decomposition

    monkeypatch.setattr(
        "orchid_ai.agents.mini_agent_decomposer.maybe_decompose",
        fake_maybe_decompose,
    )
    wrapper_mini_calls: list[dict[str, Any]] = []
    executor_mini_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mini_agent_node,
        "mini_agent_node_factory",
        _fake_mini_node_factory(wrapper_mini_calls),
    )

    config = _mini_config()
    auth = OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1")
    state = _state()

    # ── Chat path: wrapper returns the fork decision, the graph fans out. ──
    wrapper_agent = _RecordingAgent(chat_model=_StubChatModel())
    wrapper_node = _create_agent_node(wrapper_agent, agent_config=config)
    fork_update = await wrapper_node(state, with_auth(auth))
    assert "mini_agent_decisions" in fork_update
    assert wrapper_agent.calls == []  # the parent run is bypassed on a fork

    wrapper_messages = list(fork_update["messages"])
    wrapper_outcomes: dict[str, Any] = {}
    wrapper_mcp: dict[str, Any] = {}
    for sub_task in fork_update["mini_agent_decisions"]["a"]["sub_tasks"]:
        payload = {
            **state,
            "_active_mini_parent": "a",
            "_active_mini_id": sub_task["id"],
            "_active_mini_subtask": sub_task,
            "_active_mini_tool_subset": list(sub_task["resolved_tool_subset"]),
        }
        update = await mini_agent_node.mini_agent_node_factory(
            parent_config=config,
            chat_model=wrapper_agent.chat_model,
            mcp_clients=[],
        )(payload, with_auth(auth))
        wrapper_messages.extend(update["messages"])
        wrapper_outcomes.update(update["mini_agent_outcomes"])
        wrapper_mcp.update(update["mcp_context"])
    aggregator = aggregator_node_factory(parent_config=config, chat_model=wrapper_agent.chat_model)
    wrapper_aggregate = await aggregator({**state, "mini_agent_outcomes": wrapper_outcomes, "mcp_context": wrapper_mcp})

    # ── Executor path: inline gather with the same factories. ──
    monkeypatch.setattr(
        mini_agent_node,
        "mini_agent_node_factory",
        _fake_mini_node_factory(executor_mini_calls),
    )
    executor_agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(
        executor_agent,
        agent_config=config,
        inline_mini_agents=True,
    )
    executor_result = await executor.run(
        state,
        auth=auth,
        chat_id="c-1",
        message_id="m-1",
        correlation_id="corr-1",
    )

    # The parent's normal run never happens, matching the graph's fork path.
    assert executor_agent.calls == []

    # Both paths dispatched the same sub-tasks with the same tool allowlists.
    assert [(call["mini_id"], call["tool_subset"]) for call in wrapper_mini_calls] == [
        (call["mini_id"], call["tool_subset"]) for call in executor_mini_calls
    ]
    assert [call["tool_subset"] for call in executor_mini_calls] == [["tool_0"], ["tool_1"]]
    assert all(call["parent"] == "a" for call in executor_mini_calls)
    assert all(call["auth"] is auth for call in executor_mini_calls)

    # Aggregation parity: full message sequence, parent mcp_context and slots.
    wrapper_messages.extend(wrapper_aggregate["messages"])
    assert _message_view(executor_result["messages"]) == _message_view(wrapper_messages)
    assert executor_result["messages"][-1].name == "a"
    assert (
        executor_result["mcp_context"]["a"]["summary"]
        == wrapper_aggregate["mcp_context"]["a"]["summary"]
        == "aggregated answer"
    )
    assert executor_result["mcp_context"]["a"]["tool_results"] == {"result-mini_0": "value", "result-mini_1": "value"}
    assert set(executor_result["mini_agent_outcomes"]) == {"a#mini_0", "a#mini_1"}
    assert set(wrapper_outcomes) == set(executor_result["mini_agent_outcomes"])
    assert executor_result["active_agents"] == []


async def test_chat_wrapper_does_not_run_inline_mini_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.agents import mini_agent_node

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return _decomposition(count=2)

    def forbidden_factory(**kwargs: Any):
        raise AssertionError("the chat wrapper must keep its Send topology")

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)
    monkeypatch.setattr(mini_agent_node, "mini_agent_node_factory", forbidden_factory)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    auth = OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1")
    node = _create_agent_node(agent, agent_config=_mini_config())

    result = await node(_state(), with_auth(auth))

    assert "mini_agent_decisions" in result
    assert agent.calls == []


async def test_inline_mode_does_not_nest_inside_a_mini_run(monkeypatch: pytest.MonkeyPatch) -> None:
    decomposition_calls: list[dict[str, Any]] = []

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        decomposition_calls.append(kwargs)
        return _decomposition(count=2)

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)
    state = {
        **_state(),
        "_active_mini_parent": "a",
        "_active_mini_id": "mini_0",
    }

    result = await executor.run(state, auth=OrchidAuthContext(access_token="tok"), chat_id="c-1")

    assert decomposition_calls == []
    assert len(agent.calls) == 1
    assert result["active_agents"] == []


# ── Additional guardrail coverage ────────────────────────────


async def test_output_redaction_parity() -> None:
    chain = OrchidGuardrailChain([_redacting_rail("[OUTPUT REDACTED]")])
    state = _state()

    wrapper_agent = _RecordingAgent(result={"messages": [AIMessage(content="secret answer")]})
    wrapper_result = await _create_agent_node(wrapper_agent, output_guardrails=chain)(state)

    executor_agent = _RecordingAgent(result={"messages": [AIMessage(content="secret answer")]})
    executor = OrchidAgentExecutor(executor_agent, output_guardrails=chain)
    executor_result = await executor.run(state, auth=None)

    assert _message_view(wrapper_result["messages"]) == [("AIMessage", "[OUTPUT REDACTED]")]
    assert _message_view(wrapper_result["messages"]) == _message_view(executor_result["messages"])


async def test_output_guardrail_noops_when_the_agent_returns_no_messages() -> None:
    chain = OrchidGuardrailChain([_blocking_rail("blocked output")])
    state = _state()

    wrapper_agent = _RecordingAgent(result={"messages": []})
    wrapper_result = await _create_agent_node(wrapper_agent, output_guardrails=chain)(state)

    executor_agent = _RecordingAgent(result={"messages": []})
    executor = OrchidAgentExecutor(executor_agent, output_guardrails=chain)
    executor_result = await executor.run(state, auth=None)

    assert wrapper_result["messages"] == []
    assert executor_result["messages"] == []
    assert wrapper_result["active_agents"] == executor_result["active_agents"] == []


# ── Inline mini fallbacks ────────────────────────────────────


@pytest.mark.parametrize(
    "decision",
    [
        None,
        {"should_fork": True, "sub_tasks": []},
        {
            "should_fork": False,
            "sub_tasks": [
                {
                    "id": "mini_0",
                    "description": "task",
                    "instruction": "do it",
                    "allowed_tools": ["tool_0"],
                    "rationale": "independent",
                    "resolved_tool_subset": ["tool_0"],
                }
            ],
        },
    ],
    ids=["missing-decision", "no-sub-tasks", "should-fork-false"],
)
async def test_inline_mini_falls_back_without_a_valid_fork(
    monkeypatch: pytest.MonkeyPatch,
    decision: dict[str, Any] | None,
) -> None:
    from orchid_ai.agents import mini_agent_node

    update = {
        "mini_agent_decisions": {"a": decision} if decision is not None else {},
        "messages": [],
    }

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return update

    def forbidden_factory(**kwargs: Any):
        raise AssertionError("no mini may run without a valid fork decision")

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)
    monkeypatch.setattr(mini_agent_node, "mini_agent_node_factory", forbidden_factory)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)

    result = await executor.run(_state(), auth=OrchidAuthContext(access_token="tok"), chat_id="c-1")

    assert agent.calls == []
    assert result["active_agents"] == []


async def test_inline_mini_failure_does_not_leak_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.agents import mini_agent_node
    from orchid_ai.agents.mini_agent_node import _wrap_state_update

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return _decomposition(count=2)

    def outcome_factory(*, parent_config: OrchidAgentConfig, chat_model: Any, mcp_clients: Any):
        async def node(state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            mini_id = state["_active_mini_id"]
            if mini_id == "mini_1":
                outcome = {
                    "mini_id": mini_id,
                    "sub_task_description": "task 1",
                    "status": "timeout",
                    "summary": None,
                    "error": "timed out after 60s",
                    "duration_ms": 60,
                    "tool_results": {"result-mini_1": "partial"},
                }
            else:
                outcome = {
                    "mini_id": mini_id,
                    "sub_task_description": "task 0",
                    "status": "ok",
                    "summary": f"summary-{mini_id}",
                    "error": None,
                    "duration_ms": 1,
                    "tool_results": {f"result-{mini_id}": "value"},
                }
            return _wrap_state_update(parent_config.name, f"{parent_config.name}#{mini_id}", outcome)

        return node

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)
    monkeypatch.setattr(mini_agent_node, "mini_agent_node_factory", outcome_factory)

    agent = _RecordingAgent(chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)

    result = await executor.run(
        _state(),
        auth=OrchidAuthContext(access_token="tok"),
        chat_id="c-1",
        message_id="m-1",
    )

    assert agent.calls == []
    assert result["mini_agent_outcomes"]["a#mini_1"]["status"] == "timeout"
    assert result["mcp_context"]["a"]["tool_results"] == {"result-mini_0": "value"}
    assert [entry["status"] for entry in result["mcp_context"]["a"]["mini_outcomes"]] == ["ok", "timeout"]
    assert result["messages"][-1].content == "aggregated answer"
    assert result["active_agents"] == []


# ── Routed-model scope (F2) ──────────────────────────────────


class _TaggedModel:
    """Minimal duck-typed model used as the routed invocation target."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.tag)


async def test_inline_mini_aggregation_runs_inside_the_parent_routing_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchid_ai.agents import mini_agent_node
    from orchid_ai.agents.mini_agent_node import _wrap_state_update

    default_model = "test/default"
    alt_model = "test/alt"
    binding = DecisionBinding(
        spec=get_decision_point("model_routing"),
        config=OrchidDecisionPointConfig(
            model="stub",
            confidence_threshold=0.75,
            options={"models": [default_model, alt_model]},
        ),
        model=StubDecisionModel(answers={"model": {"choice": alt_model, "confidence": 0.95}}),
    )
    selector = ModelRoutingSelector(
        binding,
        default_model=default_model,
        agent_name="a",
        max_calls_per_turn=2,
    )
    alt = _TaggedModel("alt")
    routed = build_routed_chat_model(
        selector,
        default_chat_model=_TaggedModel("default"),
        model_factory=lambda model_id: alt,
    )
    seen_models: list[str] = []

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return _decomposition(count=2)

    def factory(*, parent_config: OrchidAgentConfig, chat_model: Any, mcp_clients: Any):
        async def node(state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            seen_models.append(selector.current_model)
            mini_id = state["_active_mini_id"]
            outcome = {
                "mini_id": mini_id,
                "sub_task_description": "task",
                "status": "ok",
                "summary": f"summary-{mini_id}",
                "error": None,
                "duration_ms": 1,
                "tool_results": {},
            }
            return _wrap_state_update(parent_config.name, f"{parent_config.name}#{mini_id}", outcome)

        return node

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)
    monkeypatch.setattr(mini_agent_node, "mini_agent_node_factory", factory)

    auth = OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1")
    state = _state()

    # Inline: the fork and the aggregator run inside the parent's selection.
    executor = OrchidAgentExecutor(
        _RecordingAgent(chat_model=routed),
        chat_model=routed,
        agent_config=_mini_config(),
        model_router=selector,
        inline_mini_agents=True,
    )
    executor_result = await executor.run(state, auth=auth, chat_id="c-1", message_id="m-1")

    assert seen_models == [alt_model, alt_model]
    assert executor_result["messages"][-1].content == "alt"
    # The selection is scoped to the run and restored afterwards.
    assert selector.current_model == default_model

    # Chat: the wrapper resets before the Send branches and the aggregator run.
    wrapper_agent = _RecordingAgent(chat_model=routed)
    wrapper_node = _create_agent_node(wrapper_agent, agent_config=_mini_config(), model_router=selector)
    fork_update = await wrapper_node(state, with_auth(auth))
    assert selector.current_model == default_model

    from orchid_ai.agents.mini_agent_aggregator import aggregator_node_factory

    outcomes: dict[str, Any] = {}
    mcp_context: dict[str, Any] = {}
    for sub_task in fork_update["mini_agent_decisions"]["a"]["sub_tasks"]:
        payload = {
            **state,
            "_active_mini_parent": "a",
            "_active_mini_id": sub_task["id"],
            "_active_mini_subtask": sub_task,
            "_active_mini_tool_subset": list(sub_task["resolved_tool_subset"]),
        }
        update = await mini_agent_node.mini_agent_node_factory(
            parent_config=_mini_config(),
            chat_model=routed,
            mcp_clients=[],
        )(payload, with_auth(auth))
        outcomes.update(update["mini_agent_outcomes"])
        mcp_context.update(update["mcp_context"])
    chat_aggregate = await aggregator_node_factory(parent_config=_mini_config(), chat_model=routed)(
        {**state, "mini_agent_outcomes": outcomes, "mcp_context": mcp_context}
    )

    assert chat_aggregate["messages"][-1].content == "default"
