"""Tests for src.persistence.models — ChatSession + ChatMessage."""

from __future__ import annotations

from datetime import datetime

from orchid_ai.persistence.models import ChatMessage, ChatSession


class TestChatSession:
    def test_creation(self):
        now = datetime.utcnow()
        session = ChatSession(
            id="sess-1",
            tenant_id="t1",
            user_id="u1",
            title="My Chat",
            created_at=now,
            updated_at=now,
            is_shared=True,
        )
        assert session.id == "sess-1"
        assert session.tenant_id == "t1"
        assert session.user_id == "u1"
        assert session.title == "My Chat"
        assert session.created_at == now
        assert session.updated_at == now
        assert session.is_shared is True

    def test_default_is_shared(self):
        now = datetime.utcnow()
        session = ChatSession(
            id="s",
            tenant_id="t",
            user_id="u",
            title="",
            created_at=now,
            updated_at=now,
        )
        assert session.is_shared is False


class TestChatMessage:
    def test_creation(self):
        now = datetime.utcnow()
        msg = ChatMessage(
            id="msg-1",
            chat_id="chat-1",
            role="user",
            content="Hello",
            agents_used=["basketball"],
            created_at=now,
            metadata={"key": "val"},
        )
        assert msg.id == "msg-1"
        assert msg.chat_id == "chat-1"
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.agents_used == ["basketball"]
        assert msg.metadata == {"key": "val"}

    def test_default_agents_used_and_metadata(self):
        msg = ChatMessage(id="m", chat_id="c", role="assistant", content="Hi")
        assert msg.agents_used == []
        assert msg.metadata == {}
