"""Tests for Markdown configuration — frontmatter parser + MD loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchid_ai.config.frontmatter import (
    MarkdownFile,
    compute_sha256,
    load_markdown_file,
    parse_frontmatter,
)
from orchid_ai.config.md_loader import (
    _infer_agent_name,
    _load_agents,
    _merge_agent_md,
    load_md_config,
    md_infrastructure_to_env,
)
from orchid_ai.config.schema import OrchidAgentsConfig

# ──────────────────────────────────────────────────────────────────
# Frontmatter parser
# ──────────────────────────────────────────────────────────────────


class TestParseFrontmatterBasic:
    def test_basic_frontmatter_and_body(self):
        text = "---\ntitle: Hello\n---\n\nThis is the body."
        fm, body = parse_frontmatter(text)
        assert fm == {"title": "Hello"}
        assert body == "This is the body."

    def test_empty_frontmatter(self):
        text = "---\n---\nSome body text."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Some body text."

    def test_no_frontmatter(self):
        text = "Just a plain Markdown file.\nNo frontmatter here."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == text.strip()

    def test_frontmatter_with_no_body(self):
        text = "---\nkey: value\n---"
        fm, body = parse_frontmatter(text)
        assert fm == {"key": "value"}
        assert body == ""

    def test_multiline_body(self):
        text = (
            "---\n"
            "description: Expert\n"
            "---\n"
            "\n"
            "# Title\n"
            "\n"
            "First paragraph.\n"
            "\n"
            "Second paragraph with **bold** text.\n"
            "\n"
            "> A blockquote\n"
        )
        fm, body = parse_frontmatter(text)
        assert fm == {"description": "Expert"}
        assert "# Title" in body
        assert "First paragraph" in body
        assert "> A blockquote" in body
        assert not body.startswith("\n")

    def test_bom_stripped(self):
        text = "\ufeff---\nkey: yes\n---\n\nBody after BOM."
        fm, body = parse_frontmatter(text)
        assert fm == {"key": True}  # YAML parses "yes" as boolean
        assert body == "Body after BOM."

    def test_crlf_normalised(self):
        text = "---\r\nkey: true\r\n---\r\n\r\nBody with CRLF."
        fm, body = parse_frontmatter(text)
        assert fm == {"key": True}
        assert body == "Body with CRLF."

    def test_trailing_whitespace_in_body_stripped(self):
        text = "---\nkey: v\n---\n\n  body with spaces  \n\n"
        _fm, body = parse_frontmatter(text)
        assert body == "body with spaces"


class TestComputeSHA256:
    def test_deterministic(self):
        a = compute_sha256(b"hello")
        b = compute_sha256(b"hello")
        assert a == b
        assert isinstance(a, str)
        assert len(a) == 64

    def test_different_content_different_hash(self):
        h1 = compute_sha256(b"foo")
        h2 = compute_sha256(b"bar")
        assert h1 != h2


class TestLoadMarkdownFile:
    def test_loads_and_parses(self, tmp_path):
        md_path = tmp_path / "test.md"
        md_path.write_text(
            "---\ndescription: Test Agent\n---\n\n# Agent Prompt\n\nSome prompt text.",
            encoding="utf-8",
        )
        md = load_markdown_file(md_path)
        assert isinstance(md, MarkdownFile)
        assert md.frontmatter == {"description": "Test Agent"}
        assert md.body == "# Agent Prompt\n\nSome prompt text."
        assert str(md.path) == str(md_path.resolve())
        assert len(md.sha256) == 64

    def test_no_frontmatter_file(self, tmp_path):
        md_path = tmp_path / "plain.md"
        md_path.write_text("# Just a heading", encoding="utf-8")
        md = load_markdown_file(md_path)
        assert md.frontmatter == {}
        assert md.body == "# Just a heading"
        assert len(md.sha256) == 64


# ──────────────────────────────────────────────────────────────────
# MD loader helpers
# ──────────────────────────────────────────────────────────────────


class TestInferAgentName:
    def test_from_filename_stem(self):
        assert _infer_agent_name(Path("/some/dir/basketball.md")) == "basketball"

    def test_with_dots_in_stem(self):
        assert _infer_agent_name(Path("/dir/my.agent.md")) == "my.agent"


class TestMergeAgentMD:
    def test_basic_merge(self, tmp_path):
        md_path = tmp_path / "agent.md"
        md_path.write_text(
            "---\ndescription: Test agent\nclass: mypkg.agents.test.TestAgent\n---\n# Prompt\nYou are a test agent.",
            encoding="utf-8",
        )
        md = load_markdown_file(md_path)
        data = _merge_agent_md(md)
        assert data["description"] == "Test agent"
        assert data["prompt"] == "# Prompt\nYou are a test agent."
        assert data["class"] == "mypkg.agents.test.TestAgent"

    def test_merge_with_tools(self, tmp_path):
        md_path = tmp_path / "agent.md"
        md_path.write_text(
            "---\ndescription: Tool user\ntools:\n  - tool_a\n  - tool_b\n---\nPrompt body.",
            encoding="utf-8",
        )
        md = load_markdown_file(md_path)
        data = _merge_agent_md(md)
        assert data["description"] == "Tool user"
        assert data["tools"] == ["tool_a", "tool_b"]
        assert data["prompt"] == "Prompt body."

    def test_merge_with_rag_nested(self, tmp_path):
        md_path = tmp_path / "agent.md"
        md_path.write_text(
            "---\ndescription: RAG agent\nrag:\n  enabled: true\n  k: 10\n  namespace: my_namespace\n---\nBody.",
            encoding="utf-8",
        )
        md = load_markdown_file(md_path)
        data = _merge_agent_md(md)
        assert data["description"] == "RAG agent"
        assert data["rag"] == {"enabled": True, "k": 10, "namespace": "my_namespace"}
        assert data["prompt"] == "Body."

    def test_agent_name_from_file(self):
        assert _infer_agent_name(Path("/x/y/basketball.md")) == "basketball"


# ──────────────────────────────────────────────────────────────────
# Infrastructure to env mapping
# ──────────────────────────────────────────────────────────────────


class TestMDInfrastructureToEnv:
    def test_maps_llm_model(self):
        frontmatter = {"llm": {"model": "gemini/gemini-2.5-flash"}}
        env = md_infrastructure_to_env(frontmatter)
        assert env.get("LITELLM_MODEL") == "gemini/gemini-2.5-flash"

    def test_maps_rag_settings(self):
        frontmatter = {
            "rag": {
                "vector_backend": "qdrant",
                "qdrant_url": "http://localhost:6333",
                "embedding_model": "nomic-embed-text",
            }
        }
        env = md_infrastructure_to_env(frontmatter)
        assert env.get("VECTOR_BACKEND") == "qdrant"
        assert env.get("QDRANT_URL") == "http://localhost:6333"
        assert env.get("EMBEDDING_MODEL") == "nomic-embed-text"

    def test_maps_storage(self):
        frontmatter = {
            "storage": {
                "class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
                "dsn": "sqlite:///./orchid.db",
            }
        }
        env = md_infrastructure_to_env(frontmatter)
        assert env["CHAT_STORAGE_CLASS"] == "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage"
        assert env["CHAT_DB_DSN"] == "sqlite:///./orchid.db"

    def test_unknown_keys_skipped(self):
        frontmatter = {"custom_section": {"key": "value"}}
        env = md_infrastructure_to_env(frontmatter)
        assert env == {}

    def test_agent_behaviour_keys_not_in_env(self):
        frontmatter = {"defaults": {"llm": {"model": "gemini/gemini-2.5-flash"}}}
        env = md_infrastructure_to_env(frontmatter)
        assert "LITELLM_MODEL" not in env  # defaults.llm.model is NOT an infra key

    def test_non_dict_values_skipped(self):
        frontmatter = {"version": "1", "llm": "not a dict"}
        env = md_infrastructure_to_env(frontmatter)
        assert env == {}

    def test_skip_sections_skips_storage(self):
        frontmatter = {
            "llm": {"model": "gemini/gemini-2.5-flash"},
            "storage": {
                "class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
                "dsn": "sqlite:///./orchid.db",
            },
        }
        env = md_infrastructure_to_env(frontmatter, skip_sections={"storage"})
        assert "LITELLM_MODEL" in env
        assert "CHAT_STORAGE_CLASS" not in env
        assert "CHAT_DB_DSN" not in env

    def test_skip_sections_does_not_affect_other_sections(self):
        frontmatter = {
            "llm": {"model": "gemini/gemini-2.5-flash"},
            "storage": {
                "class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
                "dsn": "sqlite:///./orchid.db",
            },
            "rag": {"embedding_model": "nomic-embed-text"},
        }
        env = md_infrastructure_to_env(frontmatter, skip_sections={"storage"})
        assert env["LITELLM_MODEL"] == "gemini/gemini-2.5-flash"
        assert env["EMBEDDING_MODEL"] == "nomic-embed-text"
        assert "CHAT_STORAGE_CLASS" not in env
        assert "CHAT_DB_DSN" not in env

    def test_skip_sections_none_skips_nothing(self):
        frontmatter = {
            "llm": {"model": "gemini/gemini-2.5-flash"},
            "storage": {
                "class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
                "dsn": "sqlite:///./orchid.db",
            },
        }
        env = md_infrastructure_to_env(frontmatter, skip_sections=None)
        assert "LITELLM_MODEL" in env
        assert "CHAT_STORAGE_CLASS" in env
        assert "CHAT_DB_DSN" in env

    def test_skip_sections_empty_set_skips_nothing(self):
        frontmatter = {"storage": {"class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage"}}
        env = md_infrastructure_to_env(frontmatter, skip_sections=set())
        assert "CHAT_STORAGE_CLASS" in env

    def test_skip_multiple_sections(self):
        frontmatter = {
            "llm": {"model": "gemini/gemini-2.5-flash"},
            "storage": {
                "class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
                "dsn": "sqlite:///./orchid.db",
            },
            "rag": {"embedding_model": "nomic-embed-text"},
        }
        env = md_infrastructure_to_env(frontmatter, skip_sections={"storage", "llm"})
        assert "CHAT_STORAGE_CLASS" not in env
        assert "CHAT_DB_DSN" not in env
        assert "LITELLM_MODEL" not in env
        assert "EMBEDDING_MODEL" in env


# ──────────────────────────────────────────────────────────────────
# MD config loader (integration)
# ──────────────────────────────────────────────────────────────────


AGENT_BASKETBALL_MD = """---
description: "Basketball statistics expert"
execution_hints:
  parallel_safe: true
rag:
  enabled: false
tools:
  - get_player_stats
  - compare_players
---

# Basketball Expert

You are a basketball statistics expert.
"""

AGENT_PSYCHOLOGIST_MD = """---
description: "Sports psychologist"
rag:
  enabled: false
tools:
  - assess_motivation
---

# Sports Psychologist

You are a sports psychologist. Be empathetic.
"""


class TestLoadMDConfig:
    @pytest.mark.asyncio
    async def test_autodetect_md_in_from_config_path(self, tmp_path):
        from orchid_ai.config.schema import OrchidAgentsConfig

        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "test_agent.md").write_text(
            "---\ndescription: Test\n---\n\nPrompt body.",
            encoding="utf-8",
        )
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )

        config, hashes = load_md_config(root, agents_dir=agents_dir)
        assert isinstance(config, OrchidAgentsConfig)
        assert "test_agent" in config.agents
        assert config.agents["test_agent"].description == "Test"
        assert config.agents["test_agent"].prompt == "Prompt body."
        agent_hash = hashes.get(str((agents_dir / "test_agent.md").resolve()))
        assert agent_hash is not None
        assert len(agent_hash) == 64

    def test_loads_simple_agent(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "basketball.md").write_text(AGENT_BASKETBALL_MD, encoding="utf-8")
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )

        config, _ = load_md_config(root, agents_dir=agents_dir)
        agent = config.agents["basketball"]
        assert agent.description == "Basketball statistics expert"
        assert "You are a basketball statistics expert" in agent.prompt
        assert agent.tools == ["get_player_stats", "compare_players"]
        assert agent.execution_hints.parallel_safe is True
        assert agent.rag.enabled is False

    def test_loads_multiple_agents(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "basketball.md").write_text(AGENT_BASKETBALL_MD, encoding="utf-8")
        (agents_dir / "psychologist.md").write_text(AGENT_PSYCHOLOGIST_MD, encoding="utf-8")
        root.write_text("---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n", encoding="utf-8")

        config, _ = load_md_config(root, agents_dir=agents_dir)
        assert "basketball" in config.agents
        assert "psychologist" in config.agents
        assert config.agents["basketball"].description == "Basketball statistics expert"
        assert config.agents["psychologist"].description == "Sports psychologist"
        assert len(config.agents) == 2

    def test_different_stems_produce_different_agents(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "dupe.md").write_text("---\ndescription: First\n---\n\nFirst body.", encoding="utf-8")
        (agents_dir / "dupe_v2.md").write_text("---\ndescription: Second\n---\n\nSecond body.", encoding="utf-8")
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )

        config, _ = load_md_config(root, agents_dir=agents_dir)
        assert "dupe" in config.agents
        assert "dupe_v2" in config.agents
        assert len(config.agents) == 2

    def test_missing_agents_dir_warns(self, caplog, tmp_path):
        root = tmp_path / "orchid.md"
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )
        config, _ = load_md_config(root, agents_dir=tmp_path / "nonexistent")
        assert len(config.agents) == 0

    def test_custom_agents_dir(self, tmp_path):
        root = tmp_path / "orchid.md"
        custom_dir = tmp_path / "custom_agents"
        custom_dir.mkdir()
        (custom_dir / "my_agent.md").write_text(
            "---\ndescription: Custom agent\n---\n\nCustom prompt.", encoding="utf-8"
        )
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )

        config, _ = load_md_config(root, agents_dir=custom_dir)
        assert "my_agent" in config.agents
        assert config.agents["my_agent"].description == "Custom agent"

    def test_agents_dir_from_frontmatter(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "my_agents"
        agents_dir.mkdir()
        (agents_dir / "agent_a.md").write_text("---\ndescription: From frontmatter\n---\n\nPrompt.", encoding="utf-8")
        root.write_text(
            "---\nagents:\n  agents_dir: my_agents\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )

        config, _ = load_md_config(root)
        assert "agent_a" in config.agents
        assert config.agents["agent_a"].description == "From frontmatter"

    def test_defaults_merge_into_agents(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "an_agent.md").write_text(
            "---\ndescription: Default merge test\n---\n\nPrompt.", encoding="utf-8"
        )
        root.write_text(
            "---\n"
            "version: '1'\n"
            "defaults:\n"
            "  llm:\n"
            "    model: ollama/llama3.2\n"
            "    temperature: 0.2\n"
            "  rag:\n"
            "    enabled: false\n"
            "---\n",
            encoding="utf-8",
        )

        config, _ = load_md_config(root, agents_dir=agents_dir)
        agent = config.agents["an_agent"]
        assert agent.llm is not None
        assert agent.llm.model == "ollama/llama3.2"
        assert agent.llm.temperature == 0.2
        assert agent.rag.enabled is False

    def test_equivalence_with_yaml(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "basketball.md").write_text(AGENT_BASKETBALL_MD, encoding="utf-8")
        root.write_text(
            "---\n"
            "version: '1'\n"
            "defaults:\n"
            "  llm:\n"
            "    model: ollama/llama3.2\n"
            "    temperature: 0.2\n"
            "  rag:\n"
            "    enabled: false\n"
            "tools:\n"
            "  get_player_stats:\n"
            "    class: examples.basketball.tools.basketball.GetPlayerStatsTool\n"
            "    description: Gets stats for an NBA player\n"
            "---\n",
            encoding="utf-8",
        )

        md_config, _ = load_md_config(root, agents_dir=agents_dir)

        yaml_fm, _ = parse_frontmatter(root.read_text(encoding="utf-8"))
        yaml_fm["agents"] = {
            "basketball": {
                "description": "Basketball statistics expert",
                "prompt": "# Basketball Expert\n\nYou are a basketball statistics expert.",
                "tools": ["get_player_stats", "compare_players"],
                "rag": {"enabled": False},
                "execution_hints": {"parallel_safe": True},
            }
        }
        yaml_config = OrchidAgentsConfig.model_validate(yaml_fm)

        assert md_config.agents["basketball"].description == yaml_config.agents["basketball"].description
        assert md_config.agents["basketball"].tools == yaml_config.agents["basketball"].tools
        assert md_config.agents["basketball"].prompt == yaml_config.agents["basketball"].prompt
        assert md_config.agents["basketball"].name == yaml_config.agents["basketball"].name

    def test_agent_name_set_from_dict_key(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "basketball.md").write_text(AGENT_BASKETBALL_MD, encoding="utf-8")
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n",
            encoding="utf-8",
        )
        config, _ = load_md_config(root, agents_dir=agents_dir)
        assert config.agents["basketball"].name == "basketball"

    def test_global_tools_in_config(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "ball.md").write_text(AGENT_BASKETBALL_MD, encoding="utf-8")
        root.write_text(
            "---\n"
            "version: '1'\n"
            "defaults:\n"
            "  rag:\n"
            "    enabled: false\n"
            "tools:\n"
            "  get_player_stats:\n"
            "    class: my.tools.GetStatsTool\n"
            "    description: Get stats\n"
            "---\n",
            encoding="utf-8",
        )
        config, _ = load_md_config(root, agents_dir=agents_dir)
        assert "get_player_stats" in config.tools
        assert config.tools["get_player_stats"].class_ == "my.tools.GetStatsTool"

    def test_file_hashes_returned(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "a.md").write_text("---\ndescription: A\n---\n\nPrompt.", encoding="utf-8")
        root.write_text("---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n---\n", encoding="utf-8")

        _, hashes = load_md_config(root, agents_dir=agents_dir)
        assert str(root.resolve()) in hashes
        assert str((agents_dir / "a.md").resolve()) in hashes
        assert all(len(h) == 64 for h in hashes.values())


# ──────────────────────────────────────────────────────────────────
# _load_agents
# ──────────────────────────────────────────────────────────────────


class TestLoadAgents:
    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        configs, hashes = _load_agents(empty_dir)
        assert configs == {}
        assert hashes == {}

    def test_missing_dir(self, tmp_path):
        configs, hashes = _load_agents(tmp_path / "no_such_dir")
        assert configs == {}
        assert hashes == {}

    def test_loads_single_agent(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "test_agent.md").write_text("---\ndescription: Test\n---\n\nPrompt body.", encoding="utf-8")
        configs, hashes = _load_agents(agents_dir)
        assert "test_agent" in configs
        assert configs["test_agent"]["description"] == "Test"
        assert configs["test_agent"]["prompt"] == "Prompt body."
        assert str((agents_dir / "test_agent.md").resolve()) in hashes


# ──────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────


class TestFrontmatterEdgeCases:
    def test_invalid_yaml_returns_empty_dict(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\n[invalid yaml\n---\n\nBody."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "Body." in body

    def test_missing_closing_delimiter_returns_empty(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\nkey: value\n\nNo closing delimiter."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "key: value" in body

    def test_frontmatter_parses_to_list_returns_empty(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\n- item1\n- item2\n---\n\nBody."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "Body." in body

    def test_frontmatter_parses_to_scalar_returns_empty(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\njust a string\n---\n\nBody."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "Body." in body

    def test_only_opening_delimiter(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\n"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        # No closing delimiter — entire text (minus opening) is treated as body
        assert body == "---"

    def test_only_opening_delimiter_with_content(self):
        from orchid_ai.config.frontmatter import parse_frontmatter

        text = "---\nsome content"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "some content" in body
