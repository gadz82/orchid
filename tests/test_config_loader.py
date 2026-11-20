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

    def test_rejects_duplicate_pipeline_definition_names(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        definition = (
            "pipelines:\n"
            "  definitions:\n"
            "    shared:\n"
            "      jobs:\n"
            "        load:\n"
            "          kind: deterministic\n"
            "          uses: app.steps.load\n"
        )
        (agents_dir / "a.yaml").write_text(definition, encoding="utf-8")
        (agents_dir / "b.yaml").write_text(definition, encoding="utf-8")

        with pytest.raises(OrchidConfigError) as excinfo:
            load_config_directory(agents_dir)

        message = str(excinfo.value)
        assert "Pipeline definition 'shared'" in message
        assert "a.yaml" in message and "b.yaml" in message

    def test_rejects_duplicate_shape_names(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        shape = "pipelines:\n  shapes:\n    Account:\n      fields:\n        id:\n          type: string\n"
        (agents_dir / "a.yaml").write_text(shape, encoding="utf-8")
        (agents_dir / "b.yaml").write_text(shape, encoding="utf-8")

        with pytest.raises(OrchidConfigError) as excinfo:
            load_config_directory(agents_dir)

        message = str(excinfo.value)
        assert "Shape 'Account'" in message
        assert "a.yaml" in message and "b.yaml" in message

    def test_rejects_duplicate_relation_names(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "a.yaml").write_text(
            "pipelines:\n"
            "  shapes:\n"
            "    Account:\n"
            "      fields:\n"
            "        id:\n"
            "          type: string\n"
            "    Usage:\n"
            "      fields:\n"
            "        id:\n"
            "          type: string\n"
            "  relations:\n"
            "    has_usage:\n"
            "      from: [Account]\n"
            "      to: [Usage]\n",
            encoding="utf-8",
        )
        (agents_dir / "b.yaml").write_text(
            "pipelines:\n  relations:\n    has_usage:\n      from: [Account]\n      to: [Usage]\n",
            encoding="utf-8",
        )

        with pytest.raises(OrchidConfigError) as excinfo:
            load_config_directory(agents_dir)

        message = str(excinfo.value)
        assert "Relation 'has_usage'" in message
        assert "a.yaml" in message and "b.yaml" in message

    def test_same_name_in_different_pipeline_sections_does_not_collide(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "pipelines.yaml").write_text(
            "pipelines:\n"
            "  shapes:\n"
            "    shared:\n"
            "      fields:\n"
            "        id:\n"
            "          type: string\n"
            "  definitions:\n"
            "    shared:\n"
            "      jobs:\n"
            "        load:\n"
            "          kind: deterministic\n"
            "          uses: app.steps.load\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert "shared" in config.pipelines.shapes
        assert "shared" in config.pipelines.definitions

    def test_valid_pipeline_directory_without_duplicates_loads(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "shapes.yaml").write_text(
            "pipelines:\n  shapes:\n    Account:\n      fields:\n        id:\n          type: string\n",
            encoding="utf-8",
        )
        (agents_dir / "pipelines.yaml").write_text(
            "pipelines:\n"
            "  definitions:\n"
            "    assess:\n"
            "      jobs:\n"
            "        load:\n"
            "          kind: deterministic\n"
            "          uses: app.steps.load\n"
            "          writes:\n"
            "            - id: account:1\n"
            "              shape: Account\n",
            encoding="utf-8",
        )

        config = load_config_directory(agents_dir)
        assert "Account" in config.pipelines.shapes
        assert "assess" in config.pipelines.definitions
