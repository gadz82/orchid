"""Tests for supervisor message filtering and history handling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.config.schema import OrchidSupervisorConfig
from orchid_ai.graph.supervisor import (
    _extract_single_agent_response,
    _filter_internal_messages,
    create_supervisor_node,
)


class TestFilterInternalMessages:
    """Verify that internal routing messages are removed."""

    def test_removes_supervisor_dispatch(self) -> None:
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="[Supervisor] Parallel dispatch: notifications"),
            AIMessage(content="Agent response"),
        ]
        result = _filter_internal_messages(messages)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Agent response"

    def test_removes_supervisor_handoff(self) -> None:
        messages = [
            AIMessage(content="[Supervisor → notifications] Continue with..."),
            AIMessage(content="[Supervisor] Sequential pipeline: a → b"),
        ]
        result = _filter_internal_messages(messages)
        assert result == []

    def test_keeps_non_supervisor_messages(self) -> None:
        messages = [
            HumanMessage(content="Query"),
            AIMessage(content="[Notifications Agent]\nHere are results"),
            AIMessage(content="Final synthesis"),
        ]
        result = _filter_internal_messages(messages)
        assert len(result) == 3  # all kept

    def test_empty_list(self) -> None:
        assert _filter_internal_messages([]) == []

    def test_custom_skip_prefixes(self) -> None:
        messages = [
            AIMessage(content="[Debug] internal log"),
            AIMessage(content="[Supervisor] routing"),
            AIMessage(content="Clean response"),
        ]
        result = _filter_internal_messages(
            messages,
            skip_prefixes=("[Debug]", "[Supervisor"),
        )
        assert len(result) == 1
        assert result[0].content == "Clean response"


class TestExtractSingleAgentResponse:
    """The fast-path helper that decides when to skip synthesis."""

    def test_returns_text_for_single_agent_turn(self):
        state = {
            "messages": [
                HumanMessage(content="What notifications exist?"),
                AIMessage(content="[Supervisor] Parallel dispatch: notifications"),
                AIMessage(content="[Notifications Agent]\nHere is the catalog: …"),
            ],
        }
        assert _extract_single_agent_response(state) == "Here is the catalog: …"

    def test_returns_none_for_multi_agent_turn(self):
        state = {
            "messages": [
                HumanMessage(content="Find courses and notify users"),
                AIMessage(content="[Supervisor] Parallel dispatch: learning, notifications"),
                AIMessage(content="[Learning Agent]\nFound 3 courses."),
                AIMessage(content="[Notifications Agent]\nDrafted 3 notifications."),
            ],
        }
        assert _extract_single_agent_response(state) is None

    def test_returns_none_for_sequential_turn(self):
        # Sequential pipeline produces multiple [X Agent] messages — synthesis
        # is needed to merge them.
        state = {
            "messages": [
                HumanMessage(content="Find then notify"),
                AIMessage(content="[Supervisor] Sequential pipeline: learning → notifications"),
                AIMessage(content="[Learning Agent]\nFound course #42."),
                AIMessage(content="[Supervisor → notifications] Continue with course #42"),
                AIMessage(content="[Notifications Agent]\nDrafted reminder."),
            ],
        }
        assert _extract_single_agent_response(state) is None

    def test_returns_none_when_no_agent_messages(self):
        state = {
            "messages": [
                HumanMessage(content="Hi"),
            ],
        }
        assert _extract_single_agent_response(state) is None

    def test_returns_none_when_agent_text_is_empty(self):
        state = {
            "messages": [
                HumanMessage(content="?"),
                AIMessage(content="[Notifications Agent]\n   "),  # whitespace-only
            ],
        }
        assert _extract_single_agent_response(state) is None

    def test_ignores_supervisor_handoffs(self):
        state = {
            "messages": [
                HumanMessage(content="?"),
                AIMessage(content="[Supervisor → notifications] handoff"),
                AIMessage(content="[Notifications Agent]\nThe answer."),
            ],
        }
        assert _extract_single_agent_response(state) == "The answer."

    def test_only_counts_messages_after_last_user_message(self):
        # Prior turns have agent outputs we must ignore — only this turn counts.
        state = {
            "messages": [
                HumanMessage(content="Old query"),
                AIMessage(content="[Notifications Agent]\nOld answer 1"),
                AIMessage(content="[Learning Agent]\nOld answer 2"),
                HumanMessage(content="New query"),
                AIMessage(content="[Notifications Agent]\nNew answer"),
            ],
        }
        assert _extract_single_agent_response(state) == "New answer"

    def test_skips_messages_with_pending_tool_calls(self):
        # An AIMessage that still has tool_calls is an intermediate
        # agentic-loop step, not a final answer.
        msg_with_tools = AIMessage(content="[Notifications Agent]\nthinking...")
        msg_with_tools.tool_calls = [{"name": "list_x", "args": {}, "id": "1"}]

        state = {
            "messages": [
                HumanMessage(content="?"),
                msg_with_tools,
                AIMessage(content="[Notifications Agent]\nFinal answer"),
            ],
        }
        assert _extract_single_agent_response(state) == "Final answer"

    def test_ignores_bare_messages_without_agent_prefix(self):
        # Bare AIMessages (e.g., the synthesis output from a prior turn
        # that got persisted, or unrelated assistant text) shouldn't
        # confuse the count.
        state = {
            "messages": [
                HumanMessage(content="?"),
                AIMessage(content="[Notifications Agent]\nThe answer"),
                AIMessage(content="A bare assistant message"),  # ignored
            ],
        }
        assert _extract_single_agent_response(state) == "The answer"


class TestSupervisorFastPath:
    """The supervisor node skips the synthesis LLM call when one agent
    produced final text."""

    @pytest.mark.asyncio
    async def test_fast_path_skips_synthesis_llm(self):
        chat_model = AsyncMock()
        # Will be called only if the fast path does NOT trigger.
        chat_model.ainvoke = AsyncMock(side_effect=AssertionError("synthesis LLM must not run"))

        node = create_supervisor_node(
            model="test-model",
            agent_descriptions={"notifications": "manages notifications"},
            chat_model=chat_model,
        )

        state = {
            "messages": [
                HumanMessage(content="?"),
                AIMessage(content="[Notifications Agent]\nHere is the answer."),
            ],
            "mcp_context": {"notifications": {}},  # has data
            "active_agents": [],
            "pending_agents": [],
        }
        result = await node(state)

        assert result["final_response"] == "Here is the answer."
        assert result["active_agents"] == []
        assert result["pending_agents"] == []
        chat_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_disabled_runs_synthesis(self):
        # When the toggle is off, the synthesis LLM still runs even for
        # single-agent turns.
        ai_response = AIMessage(content="synthesised final")
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(return_value=ai_response)

        config = OrchidSupervisorConfig(skip_synthesis_when_single_agent=False)
        node = create_supervisor_node(
            model="test-model",
            agent_descriptions={"notifications": "manages notifications"},
            chat_model=chat_model,
            supervisor_config=config,
        )

        state = {
            "messages": [
                HumanMessage(content="?"),
                AIMessage(content="[Notifications Agent]\nHere is the answer."),
            ],
            "mcp_context": {"notifications": {}},
            "active_agents": [],
            "pending_agents": [],
        }
        result = await node(state)

        chat_model.ainvoke.assert_called()
        assert result["final_response"] == "synthesised final"

    @pytest.mark.asyncio
    async def test_multi_agent_turn_still_synthesises(self):
        # Two agent outputs in the current turn → synthesis must run
        # to merge them.
        ai_response = AIMessage(content="merged answer")
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(return_value=ai_response)

        node = create_supervisor_node(
            model="test-model",
            agent_descriptions={
                "learning": "courses",
                "notifications": "alerts",
            },
            chat_model=chat_model,
        )

        state = {
            "messages": [
                HumanMessage(content="Find and notify"),
                AIMessage(content="[Learning Agent]\nCourse 42 found."),
                AIMessage(content="[Notifications Agent]\n3 reminders drafted."),
            ],
            "mcp_context": {"learning": {}, "notifications": {}},
            "active_agents": [],
            "pending_agents": [],
        }
        result = await node(state)

        chat_model.ainvoke.assert_called()
        assert result["final_response"] == "merged answer"


class TestSupervisorRoutingModel:
    """The supervisor uses a separate ``routing_chat_model`` for the
    routing + sequential-advance phases when one is supplied.  Synthesis
    keeps using the main ``chat_model``."""

    @pytest.mark.asyncio
    async def test_routing_model_used_for_routing_phase(self):
        # Routing chat model is structured-output; provide a stub that
        # records invocation and returns a routing decision.
        from orchid_ai.graph.supervisor import OrchidRoutingDecision

        decision = OrchidRoutingDecision(
            reasoning="ok",
            execution="parallel",
            agents=["notifications"],
        )

        routing_model = AsyncMock()
        routing_structured = AsyncMock()
        routing_structured.ainvoke = AsyncMock(return_value=decision)
        routing_model.with_structured_output = lambda *_args, **_kwargs: routing_structured

        # The synthesis chat model — must NOT be called during the
        # routing phase.
        synth_model = AsyncMock()
        synth_model.ainvoke = AsyncMock(side_effect=AssertionError("synthesis must not run during routing"))
        synth_model.with_structured_output = lambda *_args, **_kwargs: AsyncMock(
            ainvoke=AsyncMock(side_effect=AssertionError("synthesis must not be used to route"))
        )

        node = create_supervisor_node(
            model="big-model",
            agent_descriptions={"notifications": "alerts"},
            chat_model=synth_model,
            routing_chat_model=routing_model,
        )

        # First entry — only HumanMessage, no mcp_context yet → route phase.
        state = {
            "messages": [HumanMessage(content="What notifications are there?")],
        }
        result = await node(state)

        # Routing model was used for the structured output call
        routing_structured.ainvoke.assert_called()
        # Synthesis model was untouched
        synth_model.ainvoke.assert_not_called()
        assert result["active_agents"] == ["notifications"]

    @pytest.mark.asyncio
    async def test_routing_model_falls_back_to_chat_model_when_none(self):
        # When ``routing_chat_model`` is not supplied, routing reuses
        # ``chat_model`` (backwards compatible).
        from orchid_ai.graph.supervisor import OrchidRoutingDecision

        decision = OrchidRoutingDecision(
            reasoning="ok",
            execution="parallel",
            agents=["notifications"],
        )

        chat_model = AsyncMock()
        structured = AsyncMock()
        structured.ainvoke = AsyncMock(return_value=decision)
        chat_model.with_structured_output = lambda *_args, **_kwargs: structured

        node = create_supervisor_node(
            model="big-model",
            agent_descriptions={"notifications": "alerts"},
            chat_model=chat_model,
            # routing_chat_model omitted on purpose
        )

        state = {"messages": [HumanMessage(content="?")]}
        result = await node(state)

        structured.ainvoke.assert_called()
        assert result["active_agents"] == ["notifications"]
