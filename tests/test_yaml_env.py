"""Tests for orchid_ai.config.yaml_env — YAML-to-env mapping."""

from __future__ import annotations

import os


class TestYamlToEnvMapping:
    """Basic YAML-to-env mapping tests."""

    def test_rag_section_maps_to_env(self, monkeypatch, tmp_path):
        """Verify rag.vector_backend maps to VECTOR_BACKEND."""
        monkeypatch.delenv("VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        yml = tmp_path / "orchid.yml"
        yml.write_text(
            "rag:\n"
            "  vector_backend: qdrant\n"
            "  qdrant_url: http://localhost:6333\n"
            "  embedding_model: text-embedding-3-small\n"
        )

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml))
        assert applied == 3
        assert os.environ["VECTOR_BACKEND"] == "qdrant"
        assert os.environ["QDRANT_URL"] == "http://localhost:6333"
        assert os.environ["EMBEDDING_MODEL"] == "text-embedding-3-small"

    def test_existing_env_not_overwritten(self, monkeypatch, tmp_path):
        """Real env vars win over YAML values."""
        monkeypatch.setenv("VECTOR_BACKEND", "existing_backend")

        yml = tmp_path / "orchid.yml"
        yml.write_text("rag:\n  vector_backend: qdrant\n")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        apply_yaml_to_env(str(yml))
        assert os.environ["VECTOR_BACKEND"] == "existing_backend"


class TestCliRagSection:
    """Tests for the cli_rag: section mapping."""

    def test_cli_rag_section_maps_to_env(self, monkeypatch, tmp_path):
        """Verify cli_rag.vector_backend maps to VECTOR_BACKEND."""
        monkeypatch.delenv("VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        yml = tmp_path / "orchid.yml"
        yml.write_text("cli_rag:\n  vector_backend: chroma\n  embedding_model: ollama/nomic-embed-text\n")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml))
        assert applied == 2
        assert os.environ["VECTOR_BACKEND"] == "chroma"
        assert os.environ["EMBEDDING_MODEL"] == "ollama/nomic-embed-text"

    def test_cli_rag_all_keys(self, monkeypatch, tmp_path):
        """Verify all cli_rag keys map correctly."""
        for env_var in ["VECTOR_BACKEND", "QDRANT_URL", "EMBEDDING_MODEL", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
            monkeypatch.delenv(env_var, raising=False)

        yml = tmp_path / "orchid.yml"
        yml.write_text(
            "cli_rag:\n"
            "  vector_backend: chroma\n"
            "  qdrant_url: http://custom:6333\n"
            "  embedding_model: custom-model\n"
            "  openai_api_key: sk-test-openai\n"
            "  gemini_api_key: test-gemini-key\n"
        )

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml))
        assert applied == 5
        assert os.environ["VECTOR_BACKEND"] == "chroma"
        assert os.environ["QDRANT_URL"] == "http://custom:6333"
        assert os.environ["EMBEDDING_MODEL"] == "custom-model"
        assert os.environ["OPENAI_API_KEY"] == "sk-test-openai"
        assert os.environ["GEMINI_API_KEY"] == "test-gemini-key"

    def test_cli_rag_and_rag_coexist_first_wins(self, monkeypatch, tmp_path):
        """When both sections exist and neither is skipped, first-processed wins."""
        monkeypatch.delenv("VECTOR_BACKEND", raising=False)

        yml = tmp_path / "orchid.yml"
        # YAML iteration order: rag comes before cli_rag alphabetically,
        # but Python dict iteration preserves insertion order
        yml.write_text("rag:\n  vector_backend: qdrant\ncli_rag:\n  vector_backend: chroma\n")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        apply_yaml_to_env(str(yml))
        # First section processed wins (rag in this case)
        assert os.environ["VECTOR_BACKEND"] == "qdrant"

    def test_cli_rag_skipped_when_in_skip_sections(self, monkeypatch, tmp_path):
        """Verify skip_sections={'cli_rag'} prevents cli_rag from being applied."""
        monkeypatch.delenv("VECTOR_BACKEND", raising=False)

        yml = tmp_path / "orchid.yml"
        yml.write_text("cli_rag:\n  vector_backend: chroma\n")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml), skip_sections={"cli_rag"})
        assert applied == 0
        assert "VECTOR_BACKEND" not in os.environ

    def test_rag_skipped_when_cli_rag_present(self, monkeypatch, tmp_path):
        """When rag is skipped but cli_rag exists, cli_rag values are used."""
        monkeypatch.delenv("VECTOR_BACKEND", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

        yml = tmp_path / "orchid.yml"
        yml.write_text(
            "rag:\n"
            "  vector_backend: qdrant\n"
            "  embedding_model: gemini/gemini-embedding-001\n"
            "cli_rag:\n"
            "  vector_backend: chroma\n"
            "  embedding_model: ollama/nomic-embed-text\n"
        )

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml), skip_sections={"rag"})
        assert applied == 2
        assert os.environ["VECTOR_BACKEND"] == "chroma"
        assert os.environ["EMBEDDING_MODEL"] == "ollama/nomic-embed-text"

    def test_missing_file_returns_zero(self, monkeypatch, tmp_path):
        """Non-existent file returns 0 and doesn't raise."""
        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(tmp_path / "nonexistent.yml"))
        assert applied == 0

    def test_empty_yaml_returns_zero(self, monkeypatch, tmp_path):
        """Empty YAML file returns 0."""
        yml = tmp_path / "orchid.yml"
        yml.write_text("")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        applied = apply_yaml_to_env(str(yml))
        assert applied == 0

    def test_unknown_key_is_skipped(self, monkeypatch, tmp_path):
        """Unknown YAML keys are silently skipped."""
        yml = tmp_path / "orchid.yml"
        yml.write_text("rag:\n  unknown_key: some_value\n  vector_backend: qdrant\n")

        from orchid_ai.config.yaml_env import apply_yaml_to_env

        monkeypatch.delenv("VECTOR_BACKEND", raising=False)
        applied = apply_yaml_to_env(str(yml))
        assert applied == 1
        assert os.environ["VECTOR_BACKEND"] == "qdrant"
