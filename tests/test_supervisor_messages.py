"""Tests for supervisor message filtering and history handling."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.graph.supervisor import _filter_internal_messages


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
