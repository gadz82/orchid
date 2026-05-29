from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.memory import OrchidInMemoryConversationMemory


class TestOrchidInMemoryConversationMemory:
    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.get_conversation_summary = AsyncMock()
        storage.save_conversation_summary = AsyncMock()
        return storage

    @pytest.fixture
    def mock_chat_model(self):
        model = MagicMock()
        model.ainvoke = AsyncMock()
        return model

    @pytest.fixture
    def memory(self, mock_storage, mock_chat_model):
        return OrchidInMemoryConversationMemory(
            chat_storage=mock_storage,
            chat_model=mock_chat_model,
            structured_output=True,
        )

    @pytest.mark.asyncio
    async def test_get_running_summary(self, memory, mock_storage):
        mock_storage.get_conversation_summary.return_value = "existing summary"
        result = await memory.get_running_summary("chat-1")
        assert result == "existing summary"
        mock_storage.get_conversation_summary.assert_awaited_with("chat-1")

    @pytest.mark.asyncio
    async def test_get_running_summary_none(self, memory, mock_storage):
        mock_storage.get_conversation_summary.return_value = None
        result = await memory.get_running_summary("chat-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_empty_messages_returns_existing(self, memory):
        result = await memory.update_running_summary("chat-1", [], "existing")
        assert result == "existing"

    @pytest.mark.asyncio
    async def test_update_empty_messages_no_existing(self, memory):
        result = await memory.update_running_summary("chat-1", [], None)
        assert result == ""

    @pytest.mark.asyncio
    async def test_update_narrative_without_existing(self, mock_storage, mock_chat_model):
        mock_chat_model.ainvoke.return_value = MagicMock(content="new summary text")
        mem = OrchidInMemoryConversationMemory(
            chat_storage=mock_storage,
            chat_model=mock_chat_model,
            structured_output=False,
        )
        result = await mem.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "hello"}],
            None,
        )
        assert result == "new summary text"
        mock_storage.save_conversation_summary.assert_awaited_with("chat-1", "new summary text", 1)

    @pytest.mark.asyncio
    async def test_update_narrative_with_existing(self, mock_storage, mock_chat_model):
        mock_chat_model.ainvoke.return_value = MagicMock(content="extended summary")
        mem = OrchidInMemoryConversationMemory(
            chat_storage=mock_storage,
            chat_model=mock_chat_model,
            structured_output=False,
        )
        result = await mem.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "follow up"}],
            "previous summary",
        )
        assert result == "extended summary"

    @pytest.mark.asyncio
    async def test_update_narrative_llm_error(self, mock_storage, mock_chat_model):
        mock_chat_model.ainvoke.side_effect = RuntimeError("LLM failure")
        mem = OrchidInMemoryConversationMemory(
            chat_storage=mock_storage,
            chat_model=mock_chat_model,
            structured_output=False,
        )
        result = await mem.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "hello"}],
            "previous summary",
        )
        assert result == "previous summary"

    @pytest.mark.asyncio
    async def test_update_narrative_llm_error_no_existing(self, mock_storage, mock_chat_model):
        mock_chat_model.ainvoke.side_effect = RuntimeError("LLM failure")
        mem = OrchidInMemoryConversationMemory(
            chat_storage=mock_storage,
            chat_model=mock_chat_model,
            structured_output=False,
        )
        result = await mem.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "hello"}],
            None,
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_update_structured_from_scratch(self, memory, mock_chat_model, mock_storage):
        mock_chat_model.ainvoke.return_value = MagicMock(
            content='{"topics": ["sports"], "entities": [], "actions_taken": [], "decisions": [], "open_questions": [], "user_preferences": [], "narrative": "test", "covered_turns": 1}'
        )
        result = await memory.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "tell me about sports"}],
            None,
        )
        assert "sports" in result
        mock_storage.save_conversation_summary.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_structured_extend_existing(self, memory, mock_chat_model, mock_storage):
        mock_chat_model.ainvoke.return_value = MagicMock(
            content='{"topics": ["sports", "NBA"], "entities": [], "actions_taken": [], "decisions": [], "open_questions": [], "user_preferences": [], "narrative": "extended", "covered_turns": 1}'
        )
        existing_summary = '{"topics": ["sports"], "entities": [], "actions_taken": [], "decisions": [], "open_questions": [], "user_preferences": [], "narrative": "original", "covered_turns": 2}'
        result = await memory.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "tell me more"}],
            existing_summary,
        )
        assert "NBA" in result
        mock_storage.save_conversation_summary.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_structured_json_parse_fallback(self, memory, mock_chat_model, mock_storage):
        """When structured JSON parse fails, the raw response text is stored."""
        mock_chat_model.ainvoke.return_value = MagicMock(content="not valid json at all")
        result = await memory.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "hello"}],
            None,
        )
        assert result == "not valid json at all"

    @pytest.mark.asyncio
    async def test_update_structured_llm_error(self, memory, mock_chat_model):
        mock_chat_model.ainvoke.side_effect = RuntimeError("LLM failure")
        result = await memory.update_running_summary(
            "chat-1",
            [{"role": "user", "content": "hello"}],
            '{"topics": [], "entities": [], "actions_taken": [], "decisions": [], "open_questions": [], "user_preferences": [], "narrative": "old", "covered_turns": 1}',
        )
        assert "old" in result

    @pytest.mark.asyncio
    async def test_get_relevant_history_returns_empty(self, memory):
        result = await memory.get_relevant_history("query", "chat-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_conversation_turn_is_noop(self, memory):
        result = await memory.store_conversation_turn("chat-1", "t-1", "u-1", {"role": "user", "content": "hello"})
        assert result is None
