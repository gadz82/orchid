"""Tests for src.config.loader — YAML config loading + env interpolation."""

from __future__ import annotations

import pytest

from orchid.config.loader import _find_comment_start, _interpolate_env, load_config


# ── _find_comment_start ─────────────────────────────────────


class TestFindCommentStart:
    def test_no_comment(self):
        assert _find_comment_start("key: value") is None

    def test_comment_line(self):
        idx = _find_comment_start("key: value  # this is a comment")
        assert idx is not None
        assert idx == 12

    def test_hash_inside_single_quotes(self):
        assert _find_comment_start("key: 'value#with hash'") is None

    def test_hash_inside_double_quotes(self):
        assert _find_comment_start('key: "value#with hash"') is None


# ── _interpolate_env ────────────────────────────────────────


class TestInterpolateEnv:
    def test_replaces_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        result = _interpolate_env("url: ${MY_TEST_VAR}/api")
        assert result == "url: hello/api"

    def test_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_XYZ"):
            _interpolate_env("url: ${NONEXISTENT_VAR_XYZ}")

    def test_comment_not_interpolated(self, monkeypatch):
        monkeypatch.setenv("COMMENTED_VAR", "should_not_appear")
        result = _interpolate_env("key: value  # ${COMMENTED_VAR}")
        assert "should_not_appear" not in result
        assert "${COMMENTED_VAR}" in result


# ── load_config ─────────────────────────────────────────────


class TestLoadConfig:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_loads_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            """\
version: "1"
agents:
  test_agent:
    description: "A test agent"
    prompt: "You are a test agent"
""",
            encoding="utf-8",
        )
        config = load_config(yaml_file)
        assert "test_agent" in config.agents
        assert config.agents["test_agent"].description == "A test agent"
        assert config.agents["test_agent"].name == "test_agent"

    def test_invalid_yaml_structure(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            load_config(yaml_file)
