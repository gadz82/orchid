"""Tests for OrchidContentSourceConfig, bootstrap integration, and env-var resolution."""

from __future__ import annotations

import json
import os

import pytest

from orchid_ai.bootstrap import (
    _build_content_sources,
    _parse_content_sources_json,
    _resolve_overrides,
)
from orchid_ai.config.schema_content import OrchidContentSourceConfig
from orchid_ai.core.content import OrchidContentSource


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "CONTENT_SOURCES",
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


class TestOrchidContentSourceConfig:
    def test_valid_config(self):
        cfg = OrchidContentSourceConfig(path="/data/docs")
        assert cfg.path == "/data/docs"
        assert cfg.source == "local"
        assert cfg.file_extensions == [".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv"]
        assert cfg.metadata == {}

    def test_extra_keys_passthrough(self):
        cfg = OrchidContentSourceConfig(
            path="s3://bucket/prefix",
            source="s3",
            region="eu-west-1",
            local_fixture_path="/tmp/fixtures",
        )
        assert cfg.path == "s3://bucket/prefix"
        assert cfg.source == "s3"
        assert cfg.region == "eu-west-1"  # type: ignore[attr-defined]
        assert cfg.local_fixture_path == "/tmp/fixtures"  # type: ignore[attr-defined]

    def test_metadata_defaults(self):
        cfg = OrchidContentSourceConfig(path="/data")
        assert cfg.metadata == {}

    def test_metadata_custom(self):
        cfg = OrchidContentSourceConfig(
            path="/data",
            metadata={"category": "technical", "language": "en"},
        )
        assert cfg.metadata == {"category": "technical", "language": "en"}


class TestParseContentSourcesJson:
    def test_empty_string(self):
        assert _parse_content_sources_json("") is None

    def test_valid_json_array(self):
        result = _parse_content_sources_json('[{"path": "/data", "source": "local"}]')
        assert result == [{"path": "/data", "source": "local"}]

    def test_invalid_json(self):
        assert _parse_content_sources_json("not-json") is None

    def test_not_a_list(self):
        assert _parse_content_sources_json('{"path": "/data"}') is None


class TestBuildContentSources:
    def test_overrides_take_precedence(self):
        class DummySource(OrchidContentSource):
            async def list(self, path="", recursive=False, limit=100):
                return []

            async def get(self, path):
                raise NotImplementedError

            async def search(self, query, recursive=True, limit=10):
                return []

        override_instance = DummySource()
        result = _build_content_sources(
            [{"path": "/data"}],
            [override_instance],
        )
        assert result == [override_instance]

    def test_none_config_returns_none(self):
        assert _build_content_sources(None, None) is None

    def test_empty_config_returns_none(self):
        assert _build_content_sources([], None) is None

    def test_config_builds_local_source(self, tmp_path):
        data_dir = tmp_path / "content"
        data_dir.mkdir()
        (data_dir / "f.txt").write_text("hello")

        result = _build_content_sources(
            [{"path": str(data_dir), "source": "local"}],
            None,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0].__class__.__name__ == "LocalFileContentSource"


class TestResolveOverridesContentSources:
    def test_env_var_is_read(self, clean_env, monkeypatch):
        monkeypatch.setenv("CONTENT_SOURCES", '[{"path": "/env/docs"}]')
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
        assert ov.content_sources_json == '[{"path": "/env/docs"}]'

    def test_arg_wins_over_env(self, clean_env, monkeypatch):
        monkeypatch.setenv("CONTENT_SOURCES", '[{"path": "/env/docs"}]')
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
            content_sources_json='[{"path": "/arg/docs"}]',
            runtime_overrides=None,
        )
        assert ov.content_sources_json == '[{"path": "/arg/docs"}]'


class TestContentSourcesYamlEnv:
    def test_yaml_array_to_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CONTENT_SOURCES", raising=False)
        yml = tmp_path / "orchid.yml"
        yml.write_text("content_sources:\n  - path: /data/foo\n    source: local\n")
        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml))
        assert applied >= 1
        assert "CONTENT_SOURCES" in os.environ
        parsed = json.loads(os.environ["CONTENT_SOURCES"])
        assert parsed == [{"path": "/data/foo", "source": "local"}]

    def test_existing_env_not_overwritten(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CONTENT_SOURCES", '[{"path": "/existing"}]')
        yml = tmp_path / "orchid.yml"
        yml.write_text("content_sources:\n  - path: /data/foo\n    source: local\n")
        from orchid_ai.config.yaml_env import apply_yaml_to_env

        apply_yaml_to_env(str(yml))
        assert os.environ["CONTENT_SOURCES"] == '[{"path": "/existing"}]'
