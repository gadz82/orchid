"""Tests for OrchidAgent.compress_conversation_history (sliding-window summarization)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.agent import OrchidAgent


class _FakeLLM:
    """Fake chat model that records calls and returns a canned summary."""

    def __init__(self, response: str = "Summary of older conversation.") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> MagicMock:
        self.calls.append({"messages": messages, **kwargs})
        result = MagicMock()
        result.content = self.response
        return result


def _build_history(turn_count: int) -> list[dict[str, str]]:
    """Build a synthetic history with *turn_count* user/assistant pairs."""
    history: list[dict[str, str]] = []
    for i in range(turn_count):
        history.append({"role": "user", "content": f"User message {i}"})
        history.append({"role": "assistant", "content": f"Assistant response {i}"})
    return history


class TestCompressConversationHistory:
    """Verify the sliding-window summarization behaviour."""

    @pytest.mark.asyncio
    async def test_short_history_returned_unchanged(self) -> None:
        """History fitting within recent_turns is returned as-is (no LLM call)."""
        llm = _FakeLLM()
        history = _build_history(3)  # 6 messages

        result = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=3,
        )

        assert result == history
        assert len(llm.calls) == 0  # no LLM call

    @pytest.mark.asyncio
    async def test_exact_threshold_not_compressed(self) -> None:
        """History at exactly recent_turns * 2 is NOT compressed."""
        llm = _FakeLLM()
        history = _build_history(3)  # 6 messages, recent_turns=3 → threshold=6

        result = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=3,
        )

        assert result == history
        assert len(llm.calls) == 0

    @pytest.mark.asyncio
    async def test_long_history_compressed(self) -> None:
        """History exceeding the window is split: older → summary, recent → verbatim."""
        llm = _FakeLLM(response="The user discussed topics 0 through 6.")
        history = _build_history(10)  # 20 messages, recent_turns=3 → keep last 6

        result = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=3,
        )

        # 1 summary + 6 recent messages = 7
        assert len(result) == 7
        assert result[0]["role"] == "assistant"
        assert "[Conversation summary]" in result[0]["content"]
        assert "The user discussed topics 0 through 6." in result[0]["content"]

        # Recent turns preserved verbatim
        assert result[1] == {"role": "user", "content": "User message 7"}
        assert result[-1] == {"role": "assistant", "content": "Assistant response 9"}

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_params(self) -> None:
        """The compression LLM call uses temperature=0."""
        llm = _FakeLLM()
        history = _build_history(5)  # 10 messages

        await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=2,
        )

        assert len(llm.calls) == 1
        assert llm.calls[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_receives_older_turns_transcript(self) -> None:
        """The LLM prompt contains a transcript of only the older turns."""
        llm = _FakeLLM()
        history = _build_history(5)  # 10 messages, recent_turns=2 → older = first 6

        await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=2,
        )

        user_prompt = llm.calls[0]["messages"][1]["content"]
        # Older messages (turns 0-2) should appear in the transcript
        assert "User message 0" in user_prompt
        assert "Assistant response 2" in user_prompt
        # Recent messages (turns 3-4) should NOT appear in the transcript
        assert "User message 3" not in user_prompt

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self) -> None:
        """If the LLM call fails, fall back to just the recent turns (no summary)."""
        llm = _FakeLLM()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))  # type: ignore[method-assign]
        history = _build_history(6)  # 12 messages

        result = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=2,
        )

        # Fallback = only the 4 most recent messages
        assert len(result) == 4
        assert result[0] == {"role": "user", "content": "User message 4"}
        assert result[-1] == {"role": "assistant", "content": "Assistant response 5"}

    @pytest.mark.asyncio
    async def test_recent_turns_1(self) -> None:
        """With recent_turns=1, only the last 2 messages are kept verbatim."""
        llm = _FakeLLM(response="Summary.")
        history = _build_history(4)  # 8 messages

        result = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=llm,
            recent_turns=1,
        )

        # 1 summary + 2 recent = 3
        assert len(result) == 3
        assert result[1] == {"role": "user", "content": "User message 3"}
        assert result[2] == {"role": "assistant", "content": "Assistant response 3"}

    @pytest.mark.asyncio
    async def test_empty_history(self) -> None:
        """Empty history returns empty (no LLM call)."""
        llm = _FakeLLM()
        result = await OrchidAgent.compress_conversation_history(
            [],
            chat_model=llm,
        )
        assert result == []
        assert len(llm.calls) == 0


class TestCompressConfigSchema:
    """Verify that OrchidSupervisorConfig exposes the new compression fields."""

    def test_defaults(self) -> None:
        from orchid_ai.config.schema import OrchidSupervisorConfig

        cfg = OrchidSupervisorConfig()
        assert cfg.history_summary_enabled is True
        assert cfg.history_summary_model is None
        assert cfg.history_summary_recent_turns == 10

    def test_custom_values(self) -> None:
        from orchid_ai.config.schema import OrchidSupervisorConfig

        cfg = OrchidSupervisorConfig(
            history_summary_enabled=True,
            history_summary_model="gemini/gemini-2.5-flash-lite",
            history_summary_recent_turns=5,
        )
        assert cfg.history_summary_enabled is True
        assert cfg.history_summary_model == "gemini/gemini-2.5-flash-lite"
        assert cfg.history_summary_recent_turns == 5
