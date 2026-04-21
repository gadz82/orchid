"""Tests for src.config.schema — Pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentSkillStepConfig,
    OrchidAgentsConfig,
    OrchidBuiltinToolConfig,
    OrchidDefaultsConfig,
    ExecutionHints,
    OrchidLLMConfig,
    OrchidMCPServerConfig,
    OrchidRAGConfig,
    OrchidSupervisorConfig,
    OrchidToolConfig,
)


# ── OrchidLLMConfig ───────────────────────────────────────────────


class TestLLMConfig:
    def test_default_model(self):
        cfg = OrchidLLMConfig()
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_default_temperature(self):
        cfg = OrchidLLMConfig()
        assert cfg.temperature == 0.2


# ── OrchidRAGConfig ───────────────────────────────────────────────


class TestRAGConfig:
    def test_default_namespace(self):
        cfg = OrchidRAGConfig()
        assert cfg.namespace == ""

    def test_default_k(self):
        cfg = OrchidRAGConfig()
        assert cfg.k == 5

    def test_default_enabled(self):
        cfg = OrchidRAGConfig()
        assert cfg.enabled is True

    def test_no_dynamic_injection_field(self):
        """OrchidRAGConfig no longer has a dynamic_injection field."""
        cfg = OrchidRAGConfig()
        assert not hasattr(cfg, "dynamic_injection")


# ── OrchidMCPServerConfig ─────────────────────────────────────────


class TestMCPServerConfig:
    def test_required_fields(self):
        cfg = OrchidMCPServerConfig(name="test-mcp", url="http://localhost:3000")
        assert cfg.name == "test-mcp"
        assert cfg.url == "http://localhost:3000"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            OrchidMCPServerConfig()

    def test_default_type_and_transport(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x")
        assert cfg.type == "local"
        assert cfg.transport == "streamable_http"

    def test_wildcard_tools_string(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", tools="*")
        assert cfg.discover_all_tools is True
        assert cfg.tools == []

    def test_wildcard_tools_list(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", tools=["*"])
        assert cfg.discover_all_tools is True
        assert cfg.tools == []

    def test_wildcard_prompts(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", prompts="*")
        assert cfg.discover_all_prompts is True
        assert cfg.prompts == []

    def test_wildcard_resources(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", resources="*")
        assert cfg.discover_all_resources is True
        assert cfg.resources == []

    def test_discover_all_true_when_all_wildcarded(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", tools="*", prompts="*", resources="*")
        assert cfg.discover_all is True

    def test_discover_all_false_when_partial(self):
        cfg = OrchidMCPServerConfig(name="x", url="http://x", tools="*")
        assert cfg.discover_all is False


# ── OrchidToolConfig ──────────────────────────────────────────────


class TestToolConfig:
    def test_default_arguments(self):
        cfg = OrchidToolConfig(name="my_tool")
        assert cfg.arguments == {}

    def test_inject_to_rag_default_false(self):
        cfg = OrchidToolConfig(name="my_tool")
        assert cfg.inject_to_rag is False

    def test_inject_to_rag_explicit_true(self):
        cfg = OrchidToolConfig(name="my_tool", inject_to_rag=True)
        assert cfg.inject_to_rag is True

    def test_rag_ttl_default_none(self):
        cfg = OrchidToolConfig(name="my_tool")
        assert cfg.rag_ttl is None

    def test_rag_ttl_override(self):
        cfg = OrchidToolConfig(name="my_tool", inject_to_rag=True, rag_ttl=300)
        assert cfg.rag_ttl == 300


# ── OrchidBuiltinToolConfig ──────────────────────────────────────


class TestBuiltinToolConfig:
    def test_has_handler_and_description(self):
        cfg = OrchidBuiltinToolConfig(handler="some.module.func", description="Does stuff")
        assert cfg.handler == "some.module.func"
        assert cfg.description == "Does stuff"

    def test_rag_ttl_default_none(self):
        cfg = OrchidBuiltinToolConfig(handler="m.f")
        assert cfg.rag_ttl is None

    def test_inject_to_rag_default_false(self):
        cfg = OrchidBuiltinToolConfig(handler="some.module.func")
        assert cfg.inject_to_rag is False

    def test_inject_to_rag_explicit_true(self):
        cfg = OrchidBuiltinToolConfig(handler="some.module.func", inject_to_rag=True)
        assert cfg.inject_to_rag is True


# ── OrchidAgentSkillStepConfig ───────────────────────────────────


class TestAgentSkillStepConfig:
    def test_both_tool_and_agent_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            OrchidAgentSkillStepConfig(tool="t", agent="a")

    def test_neither_tool_nor_agent_raises(self):
        with pytest.raises(ValidationError, match="either"):
            OrchidAgentSkillStepConfig()

    def test_step_key_returns_tool_name(self):
        step = OrchidAgentSkillStepConfig(tool="my_tool")
        assert step.step_key == "my_tool"

    def test_step_key_returns_agent_name(self):
        step = OrchidAgentSkillStepConfig(agent="my_agent")
        assert step.step_key == "my_agent"


# ── ExecutionHints ──────────────────────────────────────────


class TestExecutionHints:
    def test_default_parallel_safe(self):
        cfg = ExecutionHints()
        assert cfg.parallel_safe is True


# ── OrchidSupervisorConfig ────────────────────────────────────────


class TestSupervisorConfig:
    def test_default_assistant_name(self):
        cfg = OrchidSupervisorConfig()
        assert cfg.assistant_name == "AI assistant"

    def test_default_prompt_fields_none(self):
        cfg = OrchidSupervisorConfig()
        assert cfg.routing_system_prompt is None
        assert cfg.synthesis_system_prompt is None
        assert cfg.sequential_advance_prompt is None

    def test_default_history_limits(self):
        cfg = OrchidSupervisorConfig()
        assert cfg.history_max_turns == 20
        assert cfg.history_max_chars == 1000

    def test_custom_history_limits(self):
        cfg = OrchidSupervisorConfig(history_max_turns=5, history_max_chars=500)
        assert cfg.history_max_turns == 5
        assert cfg.history_max_chars == 500


# ── OrchidAgentConfig ─────────────────────────────────────────────


class TestAgentConfig:
    def test_class_path_alias(self):
        """The YAML key 'class' maps to the 'class_path' attribute."""
        cfg = OrchidAgentConfig(
            description="desc",
            prompt="prompt",
            **{"class": "some.module.MyAgent"},
        )
        assert cfg.class_path == "some.module.MyAgent"

    def test_name_defaults_empty(self):
        cfg = OrchidAgentConfig(description="desc", prompt="prompt")
        assert cfg.name == ""

    def test_children_recursive(self):
        cfg = OrchidAgentConfig(
            description="parent",
            prompt="p",
            children={
                "child1": OrchidAgentConfig(description="child", prompt="c"),
            },
        )
        assert "child1" in cfg.children
        assert cfg.children["child1"].description == "child"


# ── OrchidAgentsConfig ────────────────────────────────────────────


class TestAgentsConfig:
    def test_apply_defaults_sets_names(self):
        cfg = OrchidAgentsConfig(
            agents={
                "basketball": OrchidAgentConfig(description="b", prompt="p"),
                "psychologist": OrchidAgentConfig(description="ps", prompt="pp"),
            },
        )
        assert cfg.agents["basketball"].name == "basketball"
        assert cfg.agents["psychologist"].name == "psychologist"

    def test_apply_defaults_merges_llm(self):
        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(llm=OrchidLLMConfig(model="ollama/llama3.2", temperature=0.5)),
            agents={
                "agent1": OrchidAgentConfig(description="d", prompt="p"),
            },
        )
        assert cfg.agents["agent1"].llm is not None
        assert cfg.agents["agent1"].llm.model == "ollama/llama3.2"
        assert cfg.agents["agent1"].llm.temperature == 0.5

    def test_explicit_llm_not_overwritten(self):
        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(llm=OrchidLLMConfig(model="ollama/llama3.2")),
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    llm=OrchidLLMConfig(model="openai/gpt-4o", temperature=0.9),
                ),
            },
        )
        assert cfg.agents["agent1"].llm.model == "openai/gpt-4o"
        assert cfg.agents["agent1"].llm.temperature == 0.9

    def test_version_defaults_to_one(self):
        cfg = OrchidAgentsConfig()
        assert cfg.version == "1"

    def test_injectable_tools_empty_by_default(self):
        """No tools have inject_to_rag=True → injectable_tools is empty."""
        cfg = OrchidAgentsConfig(
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[OrchidToolConfig(name="tool_a")],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == set()

    def test_injectable_tools_from_mcp(self):
        """MCP tools with inject_to_rag=True are collected."""
        cfg = OrchidAgentsConfig(
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[
                                OrchidToolConfig(name="tool_a", inject_to_rag=True),
                                OrchidToolConfig(name="tool_b"),
                            ],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"tool_a"}

    def test_injectable_tools_from_builtin(self):
        """Built-in tools with inject_to_rag=True are collected with builtin_ prefix."""
        cfg = OrchidAgentsConfig(
            tools={
                "format_date": OrchidBuiltinToolConfig(handler="m.f", inject_to_rag=True),
                "calc_rate": OrchidBuiltinToolConfig(handler="m.c"),
            },
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    tools=["format_date", "calc_rate"],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"builtin_format_date"}

    def test_injectable_tools_mixed(self):
        """Both MCP and built-in injectable tools are collected."""
        cfg = OrchidAgentsConfig(
            tools={
                "my_builtin": OrchidBuiltinToolConfig(handler="m.f", inject_to_rag=True),
            },
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    tools=["my_builtin"],
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[OrchidToolConfig(name="mcp_tool", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"mcp_tool", "builtin_my_builtin"}

    def test_injectable_tool_ttls_empty_when_no_ttl(self):
        """inject_to_rag=True but rag_ttl=0 → no TTL entries (cache disabled)."""
        cfg = OrchidAgentsConfig(
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[OrchidToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tool_ttls == {}

    def test_injectable_tool_ttls_from_agent_rag_ttl(self):
        """Agent-level rag_ttl propagates to all injectable tools without per-tool override."""
        cfg = OrchidAgentsConfig(
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    rag=OrchidRAGConfig(namespace="ns", rag_ttl=3600),
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[OrchidToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tool_ttls == {"tool_a": 3600}

    def test_injectable_tool_ttls_per_tool_override(self):
        """Per-tool rag_ttl overrides agent-level default."""
        cfg = OrchidAgentsConfig(
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    rag=OrchidRAGConfig(namespace="ns", rag_ttl=3600),
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[
                                OrchidToolConfig(name="tool_a", inject_to_rag=True, rag_ttl=300),
                                OrchidToolConfig(name="tool_b", inject_to_rag=True),
                            ],
                        ),
                    ],
                ),
            },
        )
        ttls = cfg.agents["agent1"].injectable_tool_ttls
        assert ttls["tool_a"] == 300
        assert ttls["tool_b"] == 3600

    def test_rag_ttl_defaults_propagation(self):
        """defaults.rag.rag_ttl propagates to agents."""
        from orchid_ai.config.schema import OrchidRAGDefaultsConfig

        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(rag=OrchidRAGDefaultsConfig(rag_ttl=7200)),
            agents={
                "agent1": OrchidAgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[OrchidToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].rag.rag_ttl == 7200
        assert cfg.agents["agent1"].injectable_tool_ttls == {"tool_a": 7200}
