"""Tests for orchid_ai.persistence.factory — build_chat_storage factory."""

from __future__ import annotations

import pytest

from orchid_ai.persistence.factory import build_chat_storage
from orchid_ai.persistence.in_memory import (
    OrchidInMemoryChatStorage,
    OrchidInMemoryMCPClientRegistrationStore,
    OrchidInMemoryMCPGatewayStateStore,
    OrchidInMemoryMCPTokenStore,
)
from orchid_ai.persistence.mcp_client_registration_factory import build_mcp_client_registration_store
from orchid_ai.persistence.mcp_gateway_state_factory import build_mcp_gateway_state_store
from orchid_ai.persistence.mcp_token_factory import build_mcp_token_store
from orchid_ai.utils import import_class


class TestImportClass:
    def test_valid_dotted_path(self):
        cls = import_class("orchid_ai.persistence.models.OrchidChatSession")
        from orchid_ai.persistence.models import OrchidChatSession

        assert cls is OrchidChatSession

    def test_invalid_path_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve class"):
            import_class("nonexistent.module.ClassName")


class TestBuildChatStorage:
    def test_invalid_class_path_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve chat storage class"):
            build_chat_storage("nonexistent.module.FakeStorage", dsn="sqlite:///test.db")

    def test_non_chat_storage_class_raises(self):
        """A class that exists but is NOT a OrchidChatStorage subclass should raise TypeError."""
        with pytest.raises(TypeError, match="not a OrchidChatStorage subclass"):
            build_chat_storage("orchid_ai.persistence.models.OrchidChatSession", dsn="sqlite:///test.db")


class TestMemorySentinel:
    """The ``"memory"`` sentinel (and empty string) select the built-in backends."""

    @pytest.mark.parametrize("sentinel", ["memory", "MEMORY", " memory ", ""])
    def test_chat_storage(self, sentinel):
        assert isinstance(build_chat_storage(sentinel, dsn=""), OrchidInMemoryChatStorage)

    @pytest.mark.parametrize("sentinel", ["memory", "MEMORY", " memory ", ""])
    def test_mcp_token_store(self, sentinel):
        assert isinstance(build_mcp_token_store(sentinel, dsn=""), OrchidInMemoryMCPTokenStore)

    @pytest.mark.parametrize("sentinel", ["memory", "MEMORY", " memory ", ""])
    def test_mcp_client_registration_store(self, sentinel):
        assert isinstance(
            build_mcp_client_registration_store(sentinel, dsn=""),
            OrchidInMemoryMCPClientRegistrationStore,
        )

    @pytest.mark.parametrize("sentinel", ["memory", "MEMORY", " memory ", ""])
    def test_mcp_gateway_state_store(self, sentinel):
        assert isinstance(build_mcp_gateway_state_store(sentinel, dsn=""), OrchidInMemoryMCPGatewayStateStore)

    def test_configured_path_never_silently_falls_back(self):
        """An explicitly configured dotted path fails loudly — no in-memory fallback."""
        with pytest.raises(ImportError, match="pip install orchid-storage-sqlite"):
            build_chat_storage("orchid_storage_sqlite.nope.Nope", dsn="x")

    def test_known_plugin_prefix_gets_pip_hint(self):
        with pytest.raises(ImportError, match="pip install orchid-storage-postgres"):
            build_chat_storage("orchid_storage_postgres.nope.Nope", dsn="x")
