"""Tests for the built-in in-memory chat storage (framework default)."""

from __future__ import annotations

from orchid_ai import Orchid
from orchid_ai.persistence.base import OrchidChatStorage
from orchid_ai.persistence.in_memory import (
    OrchidInMemoryChatStorage,
    OrchidInMemoryMCPClientRegistrationStore,
    OrchidInMemoryMCPGatewayStateStore,
    OrchidInMemoryMCPTokenStore,
)


class TestInMemoryChatStorageLifecycle:
    def test_is_orchid_chat_storage(self):
        assert isinstance(OrchidInMemoryChatStorage(), OrchidChatStorage)

    def test_constructor_accepts_factory_kwargs(self):
        """The factory contract passes ``dsn`` + ``extra_migrations_package``; both are ignored."""
        store = OrchidInMemoryChatStorage(dsn="ignored", extra_migrations_package="a.b")
        assert isinstance(store, OrchidInMemoryChatStorage)

    async def test_init_and_close_are_noops(self):
        store = OrchidInMemoryChatStorage()
        await store.init_db()
        await store.close()


class TestSessions:
    async def test_create_get_list_delete(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1", "Hello")

        assert chat.id
        assert chat.tenant_id == "t1"
        assert chat.user_id == "u1"
        assert chat.title == "Hello"
        assert chat.is_shared is False
        assert await store.get_chat(chat.id) is not None
        assert [c.id for c in await store.list_chats("t1", "u1")] == [chat.id]
        assert await store.list_chats("t1", "u2") == []
        assert await store.list_chats("t2", "u1") == []

        await store.delete_chat(chat.id)
        assert await store.get_chat(chat.id) is None
        assert await store.list_chats("t1", "u1") == []

    async def test_default_title(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        assert chat.title == "New chat"

    async def test_list_orders_by_updated_at_desc(self):
        store = OrchidInMemoryChatStorage()
        first = await store.create_chat("t1", "u1", "first")
        second = await store.create_chat("t1", "u1", "second")

        assert [c.id for c in await store.list_chats("t1", "u1")] == [second.id, first.id]

        await store.update_title(first.id, "renamed")
        assert [c.id for c in await store.list_chats("t1", "u1")] == [first.id, second.id]

    async def test_update_title(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        await store.update_title(chat.id, "Renamed")

        fetched = await store.get_chat(chat.id)
        assert fetched is not None
        assert fetched.title == "Renamed"
        assert fetched.updated_at >= chat.updated_at

    async def test_update_title_unknown_chat_is_noop(self):
        store = OrchidInMemoryChatStorage()
        await store.update_title("missing", "Nope")  # must not raise

    async def test_mark_shared(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        assert chat.is_shared is False

        await store.mark_shared(chat.id)
        fetched = await store.get_chat(chat.id)
        assert fetched is not None
        assert fetched.is_shared is True

    async def test_mark_shared_unknown_chat_is_noop(self):
        store = OrchidInMemoryChatStorage()
        await store.mark_shared("missing")  # must not raise

    async def test_returned_sessions_are_isolated_from_the_store(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1", "original")
        chat.title = "mutated"

        fetched = await store.get_chat(chat.id)
        assert fetched is not None
        assert fetched.title == "original"


class TestMessages:
    async def test_add_get_ordering_and_defaults(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")

        first = await store.add_message(chat.id, "user", "hi")
        second = await store.add_message(chat.id, "assistant", "hello", agents_used=["agent-a"], metadata={"k": "v"})

        messages = await store.get_messages(chat.id)
        assert [m.content for m in messages] == ["hi", "hello"]
        assert first.agents_used == []
        assert first.metadata == {}
        assert second.agents_used == ["agent-a"]
        assert second.metadata == {"k": "v"}
        assert second.chat_id == chat.id

    async def test_limit_and_offset(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        for i in range(5):
            await store.add_message(chat.id, "user", f"m{i}")

        window = await store.get_messages(chat.id, limit=2, offset=1)
        assert [m.content for m in window] == ["m1", "m2"]

    async def test_add_message_touches_session_updated_at(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        before = (await store.get_chat(chat.id)).updated_at

        await store.add_message(chat.id, "user", "hi")
        after = (await store.get_chat(chat.id)).updated_at
        assert after >= before

    async def test_delete_chat_cascades_messages(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        await store.add_message(chat.id, "user", "hi")

        await store.delete_chat(chat.id)
        assert await store.get_messages(chat.id) == []

    async def test_messages_for_unknown_chat_are_empty(self):
        store = OrchidInMemoryChatStorage()
        assert await store.get_messages("missing") == []

    async def test_add_message_for_unknown_chat_still_stores(self):
        """Matches the SQL engines' FK-free insert semantics for orphan chat ids."""
        store = OrchidInMemoryChatStorage()
        msg = await store.add_message("orphan", "user", "hi")
        assert [m.id for m in await store.get_messages("orphan")] == [msg.id]


class TestConversationSummaries:
    async def test_round_trip(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")

        assert await store.get_conversation_summary(chat.id) is None
        await store.save_conversation_summary(chat.id, "summary one", 1)
        assert await store.get_conversation_summary(chat.id) == "summary one"

        await store.save_conversation_summary(chat.id, "summary two", 2)
        assert await store.get_conversation_summary(chat.id) == "summary two"

    async def test_deleted_chat_drops_summary(self):
        store = OrchidInMemoryChatStorage()
        chat = await store.create_chat("t1", "u1")
        await store.save_conversation_summary(chat.id, "summary", 1)

        await store.delete_chat(chat.id)
        assert await store.get_conversation_summary(chat.id) is None


class TestFrameworkDefaultResolution:
    async def test_from_config_path_without_storage_block_uses_in_memory(self, tmp_path):
        """The framework boots with zero storage configuration — everything in-memory."""
        agents = tmp_path / "agents.yaml"
        agents.write_text(
            "agents:\n  assistant:\n    description: Test agent\n    prompt: Be helpful.\n",
            encoding="utf-8",
        )
        config = tmp_path / "orchid.yml"
        config.write_text(
            f"agents:\n  config_path: {agents}\nrag:\n  vector_backend: null\n",
            encoding="utf-8",
        )

        orchid = await Orchid.from_config_path(
            str(config),
            vector_backend="null",
            skip_yaml_sections={"rag", "storage"},
        )
        try:
            assert isinstance(orchid.chat_repo, OrchidInMemoryChatStorage)
            assert isinstance(orchid.runtime.mcp_token_store, OrchidInMemoryMCPTokenStore)
            assert isinstance(
                orchid.runtime.mcp_client_registration_store,
                OrchidInMemoryMCPClientRegistrationStore,
            )
            assert isinstance(
                orchid.runtime.mcp_gateway_client_store,
                OrchidInMemoryMCPGatewayStateStore,
            )

            chat = await orchid.chat_repo.create_chat("t1", "u1", "smoke")
            assert chat.id
        finally:
            await orchid.close()
