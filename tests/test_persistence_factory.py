"""Tests for src.persistence.factory — build_chat_storage factory."""

from __future__ import annotations

import pytest

from orchid_ai.persistence.factory import build_chat_storage
from orchid_ai.utils import import_class


class TestImportClass:
    def test_valid_dotted_path(self):
        cls = import_class("orchid_ai.persistence.models.ChatSession")
        from orchid_ai.persistence.models import ChatSession

        assert cls is ChatSession

    def test_invalid_path_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve class"):
            import_class("nonexistent.module.ClassName")


class TestBuildChatStorage:
    def test_invalid_class_path_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve chat storage class"):
            build_chat_storage("nonexistent.module.FakeStorage", dsn="sqlite:///test.db")

    def test_non_chat_storage_class_raises(self):
        """A class that exists but is NOT a ChatStorage subclass should raise TypeError."""
        with pytest.raises(TypeError, match="not a ChatStorage subclass"):
            build_chat_storage("orchid_ai.persistence.models.ChatSession", dsn="sqlite:///test.db")

    def test_forwards_extra_migrations_package(self):
        """The factory threads ``extra_migrations_package`` into the concrete backend."""
        storage = build_chat_storage(
            "orchid_ai.persistence.sqlite.SQLiteChatStorage",
            dsn=":memory:",
            extra_migrations_package="my.integrator.migrations",
        )
        # The kwarg lands on the migrator that the storage owns.
        assert storage._migrator.extra_migrations_package == "my.integrator.migrations"

    def test_defaults_extra_migrations_package_to_none(self):
        """Callers that omit the kwarg get ``None`` (no extras discovered)."""
        storage = build_chat_storage(
            "orchid_ai.persistence.sqlite.SQLiteChatStorage",
            dsn=":memory:",
        )
        assert storage._migrator.extra_migrations_package is None
