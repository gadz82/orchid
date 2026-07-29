"""Tests for config storage ABC, factory, and config model."""

from __future__ import annotations

import datetime

import pytest

from orchid_ai.config.storage import OrchidConfigStorage
from orchid_ai.config.storage_factory import build_config_storage
from orchid_ai.config.schema_storage import OrchidConfigStorageConfig
from orchid_ai.config.schema_agent import _deep_merge


class _FakeConfigStorage(OrchidConfigStorage):
    """Minimal concrete implementation of OrchidConfigStorage for testing."""

    def __init__(self) -> None:
        self._db: dict[str, dict] = {}
        self.closed = False
        self.init_called = False

    async def init_db(self) -> None:
        self.init_called = True

    async def close(self) -> None:
        self.closed = True

    async def list_configs(self) -> list[dict]:
        return list(self._db.values())

    async def get_config(self, name: str) -> dict | None:
        return self._db.get(name)

    async def upsert_config(self, name: str, config: dict) -> dict:
        now = datetime.datetime.now().isoformat()
        self._db[name] = {
            "name": name,
            "config": config,
            "created_at": now,
            "updated_at": now,
        }
        return self._db[name]

    async def patch_config(self, name: str, patch: dict) -> dict | None:
        if name not in self._db:
            return None
        existing = self._db[name]["config"]
        merged = _deep_merge(existing, patch)
        self._db[name] = {
            "name": name,
            "config": merged,
            "created_at": self._db[name]["created_at"],
            "updated_at": datetime.datetime.now().isoformat(),
        }
        return self._db[name]

    async def delete_config(self, name: str) -> None:
        self._db.pop(name, None)


class TestOrchidConfigStorageABC:
    def test_all_methods_are_abstract(self):
        import inspect

        for name, method in inspect.getmembers(OrchidConfigStorage, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            assert getattr(OrchidConfigStorage, name).__isabstractmethod__, f"{name} is not abstract"

    def test_concrete_implementation_passes_isinstance(self):
        storage = _FakeConfigStorage()
        assert isinstance(storage, OrchidConfigStorage)


class TestBuildConfigStorage:
    def test_non_existent_class_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve class"):
            build_config_storage("does.not.Exist", "dsn")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="not a subclass of OrchidConfigStorage"):
            build_config_storage("orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage", "dsn")

    def test_valid_class_returns_instance(self):
        # Plugin class lives in orchid-storage-postgres — not importable here.
        pass


class TestFakeConfigStorageIntegration:
    async def test_crud_operations(self):
        store = _FakeConfigStorage()
        await store.init_db()

        # upsert create
        row = await store.upsert_config("assistant", {"description": "A helpful agent", "prompt": "You are helpful."})
        assert row["name"] == "assistant"
        assert row["config"]["description"] == "A helpful agent"

        # upsert replace
        row2 = await store.upsert_config("assistant", {"description": "A very helpful agent"})
        assert row2["config"]["description"] == "A very helpful agent"
        assert "prompt" not in row2["config"]  # full replacement

        # list
        rows = await store.list_configs()
        assert len(rows) == 1
        assert rows[0]["name"] == "assistant"

        # get
        fetched = await store.get_config("assistant")
        assert fetched is not None
        assert fetched["config"]["description"] == "A very helpful agent"

        # get missing
        assert await store.get_config("nonexistent") is None

        # patch
        patched = await store.patch_config("assistant", {"llm": {"model": "gpt-4o"}})
        assert patched is not None
        assert patched["config"]["llm"]["model"] == "gpt-4o"
        assert patched["config"]["description"] == "A very helpful agent"  # preserved

        # patch missing returns None
        result = await store.patch_config("missing", {})
        assert result is None

        # delete
        await store.delete_config("assistant")
        assert await store.get_config("assistant") is None

        # delete idempotent (no error)
        await store.delete_config("assistant")

        await store.close()
        assert store.closed


class TestOrchidConfigStorageConfig:
    def test_defaults_disabled(self):
        cfg = OrchidConfigStorageConfig()
        assert cfg.enabled is False
        assert cfg.class_path == ""
        assert cfg.dsn == ""

    def test_full_config(self):
        cfg = OrchidConfigStorageConfig(
            enabled=True,
            class_path="orchid_storage_postgres.config_postgres.OrchidPostgresConfigStorage",
            dsn="postgresql://user:pass@host:5432/db",
        )
        assert cfg.enabled is True
        assert "OrchidPostgresConfigStorage" in cfg.class_path
        assert "5432" in cfg.dsn

    def test_model_dump(self):
        cfg = OrchidConfigStorageConfig(
            enabled=True,
            class_path="orchid_storage_postgres.config_postgres.OrchidPostgresConfigStorage",
            dsn="postgresql://user:pass@host:5432/db",
        )
        dump = cfg.model_dump()
        assert dump["enabled"] is True
        assert dump["class_path"] == "orchid_storage_postgres.config_postgres.OrchidPostgresConfigStorage"
        assert dump["dsn"] == "postgresql://user:pass@host:5432/db"
