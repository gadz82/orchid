"""Tests for OrchidAgent.extract_conversation_history."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.core.agent import OrchidAgent


class TestExtractConversationHistory:
    """Verify the framework-level conversation history extraction."""

    def test_empty_messages(self) -> None:
        assert OrchidAgent.extract_conversation_history({"messages": []}) == []

    def test_no_messages_key(self) -> None:
        assert OrchidAgent.extract_conversation_history({}) == []

    def test_single_user_message_excluded(self) -> None:
        """Last user message should be excluded — it's the current query."""
        state = {"messages": [HumanMessage(content="Hello")]}
        assert OrchidAgent.extract_conversation_history(state) == []

    def test_user_assistant_pair(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="Follow-up"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert result == [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]

    def test_supervisor_messages_skipped_by_default(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="[Supervisor] routing to agent"),
                AIMessage(content="Agent response"),
                HumanMessage(content="Follow-up"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Agent response"}

    def test_supervisor_variants_skipped(self) -> None:
        state = {
            "messages": [
                AIMessage(content="[Supervisor] Parallel dispatch: notifications"),
                AIMessage(content="[Supervisor → notifications] handoff"),
                HumanMessage(content="Current query"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert result == []

    def test_custom_skip_prefixes(self) -> None:
        state = {
            "messages": [
                AIMessage(content="[Internal] debug message"),
                HumanMessage(content="Query"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(
            state,
            skip_prefixes=("[Supervisor", "[Internal]"),
        )
        assert result == []

    def test_strip_prefixes(self) -> None:
        state = {
            "messages": [
                AIMessage(content="[MyAgent]\nActual content"),
                HumanMessage(content="Follow-up"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(
            state,
            strip_prefixes=("[MyAgent]\n",),
        )
        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "Actual content"}

    def test_strip_only_first_matching_prefix(self) -> None:
        state = {
            "messages": [
                AIMessage(content="[AgentA]\n[AgentB]\nContent"),
                HumanMessage(content="Query"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(
            state,
            strip_prefixes=("[AgentA]\n", "[AgentB]\n"),
        )
        assert len(result) == 1
        # Only first prefix stripped
        assert result[0]["content"] == "[AgentB]\nContent"

    def test_empty_content_skipped(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="  "),
                AIMessage(content=""),
                HumanMessage(content="Current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_max_turns_limits_output(self) -> None:
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"User msg {i}"))
            messages.append(AIMessage(content=f"Bot msg {i}"))
        messages.append(HumanMessage(content="Current query"))

        state = {"messages": messages}
        result = OrchidAgent.extract_conversation_history(state, max_turns=3)

        # max_turns=3 → keep last 6 messages (3 user-assistant pairs)
        assert len(result) == 6
        # Should be the most recent ones
        assert result[0]["content"] == "User msg 17"
        assert result[-1]["content"] == "Bot msg 19"

    def test_multi_turn_order_preserved(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="Turn 1"),
                AIMessage(content="Response 1"),
                HumanMessage(content="Turn 2"),
                AIMessage(content="Response 2"),
                HumanMessage(content="Turn 3"),
                AIMessage(content="Response 3"),
                HumanMessage(content="Current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert len(result) == 6
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]

    def test_last_message_is_assistant_kept(self) -> None:
        """When last message is assistant (no new user query yet), keep it."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Answer"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Question"}
        assert result[1] == {"role": "assistant", "content": "Answer"}


class TestMaxChars:
    """Tests for the max_chars truncation parameter."""

    def test_no_truncation_by_default(self) -> None:
        long_content = "x" * 5000
        state = {
            "messages": [
                AIMessage(content=long_content),
                HumanMessage(content="current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state)
        assert len(result) == 1
        assert result[0]["content"] == long_content  # not truncated

    def test_truncation_with_max_chars(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="short"),
                AIMessage(content="a" * 2000),
                HumanMessage(content="current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state, max_chars=100)
        assert len(result) == 2
        # Short message untouched
        assert result[0]["content"] == "short"
        # Long message truncated with ellipsis
        assert len(result[1]["content"]) == 101  # 100 chars + "…"
        assert result[1]["content"].endswith("…")

    def test_truncation_applies_to_user_messages(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="b" * 500),
                AIMessage(content="response"),
                HumanMessage(content="current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state, max_chars=50)
        assert len(result[0]["content"]) == 51  # 50 + "…"
        assert result[0]["content"].endswith("…")

    def test_exact_limit_not_truncated(self) -> None:
        """Message at exactly max_chars is NOT truncated."""
        content = "x" * 100
        state = {
            "messages": [
                AIMessage(content=content),
                HumanMessage(content="current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(state, max_chars=100)
        assert result[0]["content"] == content  # no ellipsis

    def test_truncation_after_strip_prefix(self) -> None:
        """Truncation is applied AFTER stripping agent prefixes."""
        state = {
            "messages": [
                AIMessage(content="[Agent]\n" + "x" * 200),
                HumanMessage(content="current"),
            ]
        }
        result = OrchidAgent.extract_conversation_history(
            state,
            max_chars=50,
            strip_prefixes=("[Agent]\n",),
        )
        # Prefix stripped first, then truncated
        assert not result[0]["content"].startswith("[Agent]")
        assert result[0]["content"].endswith("…")
        assert len(result[0]["content"]) == 51
