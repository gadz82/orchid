from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.memory_rag import OrchidRAGConversationMemory
from orchid_ai.core.repository import OrchidDocument, OrchidSearchResult


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_conversation_summary = AsyncMock()
    storage.save_conversation_summary = AsyncMock()
    return storage


@pytest.fixture
def mock_chat_model():
    model = MagicMock()
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
def mock_reader():
    reader = MagicMock()
    reader.retrieve = AsyncMock()
    return reader


@pytest.fixture
def mock_writer():
    writer = MagicMock()
    writer.upsert = AsyncMock()
    return writer


@pytest.fixture
def memory(mock_storage, mock_chat_model, mock_reader, mock_writer):
    return OrchidRAGConversationMemory(
        chat_storage=mock_storage,
        chat_model=mock_chat_model,
        reader=mock_reader,
        writer=mock_writer,
        structured_output=True,
    )


class TestOrchidRAGConversationMemory:
    @pytest.mark.asyncio
    async def test_store_conversation_turn(self, memory, mock_writer):
        await memory.store_conversation_turn(
            chat_id="chat-1",
            tenant_id="t-1",
            user_id="u-1",
            turn={"role": "user", "content": "hello world"},
        )
        assert mock_writer.upsert.call_count == 1
        _args, kwargs = mock_writer.upsert.call_args
        docs = kwargs.get("documents", _args[0] if _args else [])
        assert len(docs) == 1
        assert docs[0].page_content == "hello world"
        assert docs[0].metadata["chat_id"] == "chat-1"
        assert docs[0].metadata["tenant_id"] == "t-1"
        assert docs[0].metadata["turn_role"] == "user"

    @pytest.mark.asyncio
    async def test_store_empty_content_is_noop(self, memory, mock_writer):
        await memory.store_conversation_turn(
            chat_id="chat-1",
            tenant_id="t-1",
            user_id="u-1",
            turn={"role": "user", "content": "   "},
        )
        mock_writer.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, memory, mock_writer):
        await memory.store_conversation_turn(
            chat_id="chat-1",
            tenant_id="t-1",
            user_id="u-1",
            turn={"role": "assistant", "content": "some response"},
            metadata={"custom_field": "value"},
        )
        _args, kwargs = mock_writer.upsert.call_args
        docs = kwargs.get("documents", _args[0] if _args else [])
        assert docs[0].metadata["custom_field"] == "value"
        assert docs[0].metadata["turn_role"] == "assistant"

    @pytest.mark.asyncio
    async def test_store_writer_error_does_not_raise(self, memory, mock_writer):
        mock_writer.upsert.side_effect = RuntimeError("Qdrant down")
        await memory.store_conversation_turn(
            chat_id="chat-1",
            tenant_id="t-1",
            user_id="u-1",
            turn={"role": "user", "content": "hello"},
        )

    @pytest.mark.asyncio
    async def test_get_relevant_history(self, memory, mock_reader):
        mock_reader.retrieve.return_value = [
            OrchidSearchResult(
                document=OrchidDocument(id="d1", page_content="relevant turn", metadata={}),
                score=0.85,
            )
        ]
        result = await memory.get_relevant_history(
            query="test query",
            chat_id="chat-1",
            k=3,
            tenant_id="t-1",
            user_id="u-1",
            similarity_threshold=0.5,
        )
        assert len(result) == 1
        assert result[0]["content"] == "relevant turn"
        assert result[0]["role"] == "assistant"
        mock_reader.retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_relevant_history_below_threshold(self, memory, mock_reader):
        mock_reader.retrieve.return_value = [
            OrchidSearchResult(
                document=OrchidDocument(id="d1", page_content="low relevance", metadata={}),
                score=0.3,
            )
        ]
        result = await memory.get_relevant_history(
            query="test",
            chat_id="chat-1",
            k=5,
            tenant_id="t-1",
            user_id="u-1",
            similarity_threshold=0.5,
        )
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_relevant_history_retrieve_error(self, memory, mock_reader):
        mock_reader.retrieve.side_effect = RuntimeError("Query failed")
        result = await memory.get_relevant_history(
            query="test",
            chat_id="chat-1",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_relevant_history_empty_results(self, memory, mock_reader):
        mock_reader.retrieve.return_value = []
        result = await memory.get_relevant_history(
            query="test",
            chat_id="chat-1",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_relevant_history_merged_no_rag_turns(self, memory, mock_reader):
        mock_reader.retrieve.return_value = []
        verbatim = [{"role": "user", "content": "recent message"}]
        result = await memory.get_relevant_history_merged(
            query="test",
            chat_id="chat-1",
            recent_verbatim=verbatim,
            tenant_id="t-1",
            user_id="u-1",
        )
        assert result == verbatim

    @pytest.mark.asyncio
    async def test_get_relevant_history_merged_with_dedup(self, memory, mock_reader):
        mock_reader.retrieve.return_value = [
            OrchidSearchResult(
                document=OrchidDocument(id="d1", page_content="hello world", metadata={}),
                score=0.9,
            )
        ]
        verbatim = [{"role": "user", "content": "hello world"}]
        result = await memory.get_relevant_history_merged(
            query="test",
            chat_id="chat-1",
            recent_verbatim=verbatim,
            tenant_id="t-1",
            user_id="u-1",
        )
        assert len(result) == 1  # deduplicated

    @pytest.mark.asyncio
    async def test_get_relevant_history_merged_no_dedup(self, memory, mock_reader):
        mock_reader.retrieve.return_value = [
            OrchidSearchResult(
                document=OrchidDocument(id="d1", page_content="unique rag turn", metadata={}),
                score=0.9,
            )
        ]
        verbatim = [{"role": "user", "content": "different verbatim turn"}]
        result = await memory.get_relevant_history_merged(
            query="test",
            chat_id="chat-1",
            recent_verbatim=verbatim,
        )
        assert len(result) == 2
        assert result[0]["content"] == "unique rag turn"
        assert result[1]["content"] == "different verbatim turn"

    @pytest.mark.asyncio
    async def test_inherits_get_running_summary(self, memory, mock_storage):
        mock_storage.get_conversation_summary.return_value = "summary from storage"
        result = await memory.get_running_summary("chat-1")
        assert result == "summary from storage"

    @pytest.mark.asyncio
    async def test_inherits_get_relevant_history_empty(self, memory):
        result = await memory.get_relevant_history("query", "chat-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_turn_with_missing_content_key(self, memory, mock_writer):
        await memory.store_conversation_turn(
            chat_id="chat-1",
            tenant_id="t-1",
            user_id="u-1",
            turn={"role": "user"},
        )
        mock_writer.upsert.assert_not_called()
