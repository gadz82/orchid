"""Tests for orchid_ai.config.loader — single-file and directory loading."""

from __future__ import annotations

import pytest

from orchid_ai.config.errors import OrchidConfigError
from orchid_ai.config.loader import load_config, load_config_directory
from orchid_ai.config.schema import OrchidAgentsConfig


class TestLoadConfigDirectory:
    """Load a directory of YAML files and merge them into one config."""

    def test_loads_agents_from_directory(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        (agents_dir / "_shared.yaml").write_text(
            "version: '1'\ndefaults:\n  llm:\n    model: gemini/gemini-flash-latest\n",
            encoding="utf-8",
        )
        (agents_dir / "basketball.yaml").write_text(
            "agents:\n  basketball:\n    description: Basketball expert\n    prompt: You know basketball.\n",
            encoding="utf-8",
        )
        (agents_dir / "psychologist.yaml").write_text(
            "agents:\n  psychologist:\n    description: Sports psychologist\n    prompt: You know sports psychology.\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert isinstance(config, OrchidAgentsConfig)
        assert set(config.agents.keys()) == {"basketball", "psychologist"}
        assert config.agents["basketball"].llm.model == "gemini/gemini-flash-latest"
        assert config.agents["psychologist"].llm.model == "gemini/gemini-flash-latest"

    def test_load_config_accepts_directory(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent.yaml").write_text(
            "agents:\n  alpha:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        config = load_config(agents_dir)
        assert "alpha" in config.agents

    def test_rejects_duplicate_agent_names(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "a.yaml").write_text(
            "agents:\n  duplicate:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )
        (agents_dir / "b.yaml").write_text(
            "agents:\n  duplicate:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        with pytest.raises(OrchidConfigError, match="Agent 'duplicate' is defined in both"):
            load_config_directory(agents_dir)

    def test_empty_directory_raises(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        with pytest.raises(OrchidConfigError, match="No YAML files found"):
            load_config_directory(agents_dir)

    def test_env_interpolation_in_directory_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHARED_MODEL", "ollama/llama3.2")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "shared.yaml").write_text(
            "defaults:\n  llm:\n    model: ${SHARED_MODEL}\n",
            encoding="utf-8",
        )
        (agents_dir / "agent.yaml").write_text(
            "agents:\n  alpha:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert config.agents["alpha"].llm.model == "ollama/llama3.2"

    def test_top_level_keys_merge_across_files(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "skills.yaml").write_text(
            "skills:\n  end_to_end:\n    description: Trace a flow\n    steps: []\n",
            encoding="utf-8",
        )
        (agents_dir / "agents.yaml").write_text(
            "agents:\n  alpha:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert "end_to_end" in config.skills
        assert "alpha" in config.agents

    def test_loads_both_yaml_and_yml_extensions(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "one.yaml").write_text(
            "agents:\n  one:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )
        (agents_dir / "two.yml").write_text(
            "agents:\n  two:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert {"one", "two"} <= set(config.agents.keys())

    def test_load_config_directory_does_not_mutate_input_path(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent.yaml").write_text(
            "agents:\n  alpha:\n    description: d\n    prompt: p\n",
            encoding="utf-8",
        )

        original = str(agents_dir)
        load_config_directory(agents_dir)
        assert str(agents_dir) == original
