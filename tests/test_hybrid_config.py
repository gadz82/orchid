"""Tests for hybrid config — YAML root (orchid.yml) + MD agents (agents/*.md)."""

from __future__ import annotations

from orchid_ai.config.md_loader import _load_agents, build_config_data_from_yaml
from orchid_ai.config.schema import OrchidAgentsConfig

AGENT_BALL_MD = """---
description: "Basketball expert"
tools:
  - get_player_stats
---

# Basketball Expert

You are a basketball stats expert.
"""

AGENT_PSYCH_MD = """---
description: "Psychologist"
tools:
  - assess_motivation
---

# Psychologist

You are a sports psychologist.
"""

_AGENT_BEHAVIOUR_FIELDS = frozenset(OrchidAgentsConfig.model_fields.keys())


class TestHybridConfigBuilding:
    """Test the config-building part of hybrid mode without full runtime init."""

    def test_hybrid_builds_valid_config(self, tmp_path):
        import yaml

        # Setup orchid.yml with both infra and agent-behaviour keys
        root = tmp_path / "orchid.yml"
        root.write_text(
            "llm:\n"
            "  model: ollama/llama3.2\n"
            "auth:\n"
            "  dev_bypass: true\n"
            "\n"
            "version: '1'\n"
            "defaults:\n"
            "  rag:\n"
            "    enabled: false\n"
            "tools:\n"
            "  get_player_stats:\n"
            "    handler: my.tools.get_stats\n"
            "    description: Get stats\n"
            "supervisor:\n"
            "  assistant_name: Test AI\n",
            encoding="utf-8",
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "basketball.md").write_text(AGENT_BALL_MD, encoding="utf-8")
        (agents_dir / "psychologist.md").write_text(AGENT_PSYCH_MD, encoding="utf-8")

        # Read YAML for top-level config
        with open(root, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Load agents from MD
        agent_configs, _ = _load_agents(agents_dir)

        # Build config data using the shared helper
        config_data = build_config_data_from_yaml(yaml_data, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

        # Validate
        config = OrchidAgentsConfig.model_validate(config_data)

        assert "basketball" in config.agents
        assert "psychologist" in config.agents
        assert config.agents["basketball"].description == "Basketball expert"
        assert "You are a basketball stats expert" in config.agents["basketball"].prompt
        assert config.agents["basketball"].tools == ["get_player_stats"]
        assert config.agents["psychologist"].description == "Psychologist"
        assert "get_player_stats" in config.tools
        assert config.supervisor.assistant_name == "Test AI"

    def test_hybrid_with_no_top_level_agents_key(self, tmp_path):
        import yaml

        root = tmp_path / "orchid.yml"
        root.write_text(
            "version: '1'\ndefaults:\n  rag:\n    enabled: false\n",
            encoding="utf-8",
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "my_agent.md").write_text("---\ndescription: MyAgent\n---\n\nPrompt.", encoding="utf-8")

        with open(root, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        agent_configs, _ = _load_agents(agents_dir)
        config_data = build_config_data_from_yaml(yaml_data, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

        config = OrchidAgentsConfig.model_validate(config_data)
        assert "my_agent" in config.agents
        assert config.agents["my_agent"].description == "MyAgent"

    def test_hybrid_defaults_merge(self, tmp_path):
        import yaml

        root = tmp_path / "orchid.yml"
        root.write_text(
            "version: '1'\n"
            "defaults:\n"
            "  llm:\n"
            "    model: ollama/llama3.2\n"
            "    temperature: 0.2\n"
            "  rag:\n"
            "    enabled: false\n",
            encoding="utf-8",
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent.md").write_text("---\ndescription: Test\n---\n\nPrompt.", encoding="utf-8")

        with open(root, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        agent_configs, _ = _load_agents(agents_dir)
        config_data = build_config_data_from_yaml(yaml_data, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

        config = OrchidAgentsConfig.model_validate(config_data)
        agent = config.agents["agent"]
        assert agent.llm is not None
        assert agent.llm.model == "ollama/llama3.2"
        assert agent.llm.temperature == 0.2
        assert agent.rag.enabled is False
