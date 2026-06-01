"""Tests for the focused helpers inside ``orchid_ai.bootstrap``.

These exercise ``_resolve_overrides``, ``_prepare_reader``, and
``_run_startup_hook`` in isolation — the tests for the full
``build_runtime`` orchestration live in ``test_client.py`` /
``test_bootstrap_validation.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orchid_ai.bootstrap import (
    _prepare_reader,
    _resolve_overrides,
    _run_startup_hook,
)
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidRAGConfig


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "AGENTS_CONFIG_PATH",
        "LITELLM_MODEL",
        "VECTOR_BACKEND",
        "QDRANT_URL",
        "EMBEDDING_MODEL",
        "CHAT_STORAGE_CLASS",
        "CHAT_DB_DSN",
        "CHAT_EXTRA_MIGRATIONS_PACKAGE",
        "MCP_TOKEN_STORE_CLASS",
        "MCP_TOKEN_STORE_DSN",
        "CHECKPOINTER_TYPE",
        "CHECKPOINTER_DSN",
        "STARTUP_HOOK",
    ):
        monkeypatch.delenv(var, raising=False)


class TestResolveOverrides:
    def test_arg_wins_over_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("LITELLM_MODEL", "from-env")
        ov = _resolve_overrides(
            agents_config_path="",
            model="from-arg",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.model == "from-arg"

    def test_env_wins_over_default(self, clean_env, monkeypatch):
        monkeypatch.setenv("LITELLM_MODEL", "from-env")
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.model == "from-env"

    def test_defaults_apply(self, clean_env):
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.model == "ollama/llama3.2"
        assert ov.storage_dsn == "~/.orchid/chats.db"
        assert ov.token_store_dsn == ov.storage_dsn  # same-file default

    def test_extra_migrations_arg_wins_over_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("CHAT_EXTRA_MIGRATIONS_PACKAGE", "from.env.pkg")
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package="from.arg.pkg",
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.extra_migrations_package == "from.arg.pkg"

    def test_extra_migrations_env_fallback(self, clean_env, monkeypatch):
        monkeypatch.setenv("CHAT_EXTRA_MIGRATIONS_PACKAGE", "from.env.pkg")
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.extra_migrations_package == "from.env.pkg"

    def test_extra_migrations_defaults_to_none(self, clean_env):
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=None,
        )
        assert ov.extra_migrations_package is None

    def test_runtime_overrides_is_copied(self, clean_env):
        original = {"reader": "foo"}
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides=original,
        )
        ov.runtime_overrides["reader"] = "mutated"
        assert original == {"reader": "foo"}  # caller's dict untouched


class TestPrepareReader:
    @pytest.mark.asyncio
    async def test_caller_provided_reader_wins(self, clean_env):
        custom = MagicMock()
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides={"reader": custom},
        )
        # Empty agents → no namespaces → no ensure/warn
        empty_config = OrchidAgentsConfig(agents={})
        reader = await _prepare_reader(ov, empty_config)
        assert reader is custom

    @pytest.mark.asyncio
    async def test_warns_when_agents_need_admin_reader(self, clean_env, caplog):
        # A plain mock is not a OrchidVectorStoreAdmin.
        plain = MagicMock(spec=object)
        ov = _resolve_overrides(
            agents_config_path="",
            model="",
            vector_backend="",
            qdrant_url="",
            embedding_model="",
            chat_storage_class="",
            chat_db_dsn="",
            chat_extra_migrations_package=None,
            mcp_token_store_class="",
            mcp_token_store_dsn="",
            mcp_client_registration_store_class="",
            mcp_client_registration_store_dsn="",
            mcp_gateway_state_store_class="",
            mcp_gateway_state_store_dsn="",
            checkpointer_type="",
            checkpointer_dsn="",
            startup_hook="",
            runtime_overrides={"reader": plain},
        )
        config = OrchidAgentsConfig(
            agents={
                "a": OrchidAgentConfig(
                    description="a",
                    prompt="p",
                    rag=OrchidRAGConfig(enabled=True, namespace="foo"),
                )
            }
        )
        import logging

        with caplog.at_level(logging.WARNING):
            await _prepare_reader(ov, config)
        assert any("does not implement" in r.message for r in caplog.records)


class TestRunStartupHook:
    @pytest.mark.asyncio
    async def test_no_hook_path_is_noop(self, clean_env):
        # When hook_path is empty, nothing is imported or called.
        await _run_startup_hook("", reader=object(), runtime=object(), extra_kwargs=None)

    @pytest.mark.asyncio
    async def test_hook_receives_reader_runtime_and_extra_kwargs(self, clean_env, tmp_path, monkeypatch):
        calls: list[dict] = []

        async def my_hook(**kwargs):
            calls.append(kwargs)

        # Stash the hook on a temporary module so `import_class` can resolve it.
        import sys
        import types

        mod = types.ModuleType("fake_hook_module")
        mod.entry = my_hook
        sys.modules["fake_hook_module"] = mod

        runtime_sentinel = object()
        try:
            await _run_startup_hook(
                "fake_hook_module.entry",
                reader="reader-sentinel",
                runtime=runtime_sentinel,
                extra_kwargs={"settings": "s", "extra": 1},
            )
        finally:
            del sys.modules["fake_hook_module"]

        # ``runtime`` flows alongside ``reader`` so hooks can configure
        # post-construction collaborators (e.g. graph_store).
        assert calls == [
            {
                "reader": "reader-sentinel",
                "runtime": runtime_sentinel,
                "settings": "s",
                "extra": 1,
                "agents_config": None,
            }
        ]
