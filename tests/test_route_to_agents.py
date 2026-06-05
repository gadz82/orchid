"""Regression tests for ``route_to_agents`` dispatch precedence."""

from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Send

from orchid_ai.graph.supervisor import route_to_agents


def test_active_agents_dispatch_despite_stale_final_response():
    """A ``final_response`` left in a resumed checkpoint must NOT abort a
    fresh dispatch.

    Regression: with a persisting checkpointer, a prior turn's
    ``final_response`` survived into the next turn's state. When the
    supervisor then activated an agent, ``route_to_agents`` saw the stale
    ``final_response`` first and short-circuited to END — the agent never
    ran and the stale answer was replayed.
    """
    state = {
        "final_response": "stale answer from a previous turn",
        "active_agents": ["api_designer"],
        "execution_mode": "parallel",
    }
    result = route_to_agents(state)
    assert isinstance(result, list)
    assert [s.node for s in result] == ["api_designer_agent"]


def test_active_agents_sequential_dispatch_despite_stale_final_response():
    state = {
        "final_response": "stale",
        "active_agents": ["a", "b"],
        "execution_mode": "sequential",
    }
    assert route_to_agents(state) == "a_agent"


def test_direct_response_without_agents_still_ends():
    state = {"final_response": "direct answer", "active_agents": []}
    assert route_to_agents(state) == END


def test_pending_agents_reenter_supervisor():
    state = {"active_agents": [], "pending_agents": ["next"]}
    assert route_to_agents(state) == "supervisor"


def test_send_payload_is_the_state():
    state = {"active_agents": ["x"], "execution_mode": "parallel"}
    result = route_to_agents(state)
    assert isinstance(result[0], Send)
    assert result[0].arg is state


# ── Turn-aware synthesise/route gate (checkpointer-safe) ─────


def test_fresh_turn_with_stale_mcp_context_routes_not_synthesises():
    """Stale agent output from a prior turn (kept by the checkpointer) must
    NOT make the supervisor skip routing on a fresh user turn.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from orchid_ai.graph.supervisor import _current_turn_has_agent_output

    state = {
        "messages": [
            HumanMessage(content="previous q"),
            AIMessage(content="[api_designer Agent]\nprev answer"),
            HumanMessage(content="new question this turn"),
        ],
    }
    # No agent output AFTER the latest human → supervisor must route.
    assert _current_turn_has_agent_output(state) is False


def test_agents_ran_this_turn_is_detected():
    from langchain_core.messages import AIMessage, HumanMessage

    from orchid_ai.graph.supervisor import _current_turn_has_agent_output

    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="[Supervisor] Parallel dispatch: api_designer"),
            AIMessage(content="[api_designer Agent]\nhere is the design"),
        ],
    }
    assert _current_turn_has_agent_output(state) is True


def test_supervisor_banner_only_is_not_agent_output():
    from langchain_core.messages import AIMessage, HumanMessage

    from orchid_ai.graph.supervisor import _current_turn_has_agent_output

    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="[Supervisor] Parallel dispatch: x"),
        ],
    }
    assert _current_turn_has_agent_output(state) is False
