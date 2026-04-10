"""Tests for src.config.schema — Pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema import (
    AgentConfig,
    AgentSkillStepConfig,
    AgentsConfig,
    BuiltinToolConfig,
    DefaultsConfig,
    ExecutionHints,
    LLMConfig,
    MCPServerConfig,
    RAGConfig,
    SupervisorConfig,
    ToolConfig,
)


# ── LLMConfig ───────────────────────────────────────────────


class TestLLMConfig:
    def test_default_model(self):
        cfg = LLMConfig()
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_default_temperature(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.2


# ── RAGConfig ───────────────────────────────────────────────


class TestRAGConfig:
    def test_default_namespace(self):
        cfg = RAGConfig()
        assert cfg.namespace == ""

    def test_default_k(self):
        cfg = RAGConfig()
        assert cfg.k == 5

    def test_default_enabled(self):
        cfg = RAGConfig()
        assert cfg.enabled is True

    def test_no_dynamic_injection_field(self):
        """RAGConfig no longer has a dynamic_injection field."""
        cfg = RAGConfig()
        assert not hasattr(cfg, "dynamic_injection")


# ── MCPServerConfig ─────────────────────────────────────────


class TestMCPServerConfig:
    def test_required_fields(self):
        cfg = MCPServerConfig(name="test-mcp", url="http://localhost:3000")
        assert cfg.name == "test-mcp"
        assert cfg.url == "http://localhost:3000"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            MCPServerConfig()

    def test_default_type_and_transport(self):
        cfg = MCPServerConfig(name="x", url="http://x")
        assert cfg.type == "local"
        assert cfg.transport == "streamable_http"

    def test_wildcard_tools_string(self):
        cfg = MCPServerConfig(name="x", url="http://x", tools="*")
        assert cfg.discover_all_tools is True
        assert cfg.tools == []

    def test_wildcard_tools_list(self):
        cfg = MCPServerConfig(name="x", url="http://x", tools=["*"])
        assert cfg.discover_all_tools is True
        assert cfg.tools == []

    def test_wildcard_prompts(self):
        cfg = MCPServerConfig(name="x", url="http://x", prompts="*")
        assert cfg.discover_all_prompts is True
        assert cfg.prompts == []

    def test_wildcard_resources(self):
        cfg = MCPServerConfig(name="x", url="http://x", resources="*")
        assert cfg.discover_all_resources is True
        assert cfg.resources == []

    def test_discover_all_true_when_all_wildcarded(self):
        cfg = MCPServerConfig(name="x", url="http://x", tools="*", prompts="*", resources="*")
        assert cfg.discover_all is True

    def test_discover_all_false_when_partial(self):
        cfg = MCPServerConfig(name="x", url="http://x", tools="*")
        assert cfg.discover_all is False


# ── ToolConfig ──────────────────────────────────────────────


class TestToolConfig:
    def test_default_arguments(self):
        cfg = ToolConfig(name="my_tool")
        assert cfg.arguments == {}

    def test_inject_to_rag_default_false(self):
        cfg = ToolConfig(name="my_tool")
        assert cfg.inject_to_rag is False

    def test_inject_to_rag_explicit_true(self):
        cfg = ToolConfig(name="my_tool", inject_to_rag=True)
        assert cfg.inject_to_rag is True

    def test_rag_ttl_default_none(self):
        cfg = ToolConfig(name="my_tool")
        assert cfg.rag_ttl is None

    def test_rag_ttl_override(self):
        cfg = ToolConfig(name="my_tool", inject_to_rag=True, rag_ttl=300)
        assert cfg.rag_ttl == 300


# ── BuiltinToolConfig ──────────────────────────────────────


class TestBuiltinToolConfig:
    def test_has_handler_and_description(self):
        cfg = BuiltinToolConfig(handler="some.module.func", description="Does stuff")
        assert cfg.handler == "some.module.func"
        assert cfg.description == "Does stuff"

    def test_rag_ttl_default_none(self):
        cfg = BuiltinToolConfig(handler="m.f")
        assert cfg.rag_ttl is None

    def test_inject_to_rag_default_false(self):
        cfg = BuiltinToolConfig(handler="some.module.func")
        assert cfg.inject_to_rag is False

    def test_inject_to_rag_explicit_true(self):
        cfg = BuiltinToolConfig(handler="some.module.func", inject_to_rag=True)
        assert cfg.inject_to_rag is True


# ── AgentSkillStepConfig ───────────────────────────────────


class TestAgentSkillStepConfig:
    def test_both_tool_and_agent_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            AgentSkillStepConfig(tool="t", agent="a")

    def test_neither_tool_nor_agent_raises(self):
        with pytest.raises(ValidationError, match="either"):
            AgentSkillStepConfig()

    def test_step_key_returns_tool_name(self):
        step = AgentSkillStepConfig(tool="my_tool")
        assert step.step_key == "my_tool"

    def test_step_key_returns_agent_name(self):
        step = AgentSkillStepConfig(agent="my_agent")
        assert step.step_key == "my_agent"


# ── ExecutionHints ──────────────────────────────────────────


class TestExecutionHints:
    def test_default_parallel_safe(self):
        cfg = ExecutionHints()
        assert cfg.parallel_safe is True


# ── SupervisorConfig ────────────────────────────────────────


class TestSupervisorConfig:
    def test_default_assistant_name(self):
        cfg = SupervisorConfig()
        assert cfg.assistant_name == "AI assistant"

    def test_default_prompt_fields_none(self):
        cfg = SupervisorConfig()
        assert cfg.routing_system_prompt is None
        assert cfg.synthesis_system_prompt is None
        assert cfg.sequential_advance_prompt is None


# ── AgentConfig ─────────────────────────────────────────────


class TestAgentConfig:
    def test_class_path_alias(self):
        """The YAML key 'class' maps to the 'class_path' attribute."""
        cfg = AgentConfig(
            description="desc",
            prompt="prompt",
            **{"class": "some.module.MyAgent"},
        )
        assert cfg.class_path == "some.module.MyAgent"

    def test_name_defaults_empty(self):
        cfg = AgentConfig(description="desc", prompt="prompt")
        assert cfg.name == ""

    def test_children_recursive(self):
        cfg = AgentConfig(
            description="parent",
            prompt="p",
            children={
                "child1": AgentConfig(description="child", prompt="c"),
            },
        )
        assert "child1" in cfg.children
        assert cfg.children["child1"].description == "child"


# ── AgentsConfig ────────────────────────────────────────────


class TestAgentsConfig:
    def test_apply_defaults_sets_names(self):
        cfg = AgentsConfig(
            agents={
                "basketball": AgentConfig(description="b", prompt="p"),
                "psychologist": AgentConfig(description="ps", prompt="pp"),
            },
        )
        assert cfg.agents["basketball"].name == "basketball"
        assert cfg.agents["psychologist"].name == "psychologist"

    def test_apply_defaults_merges_llm(self):
        cfg = AgentsConfig(
            defaults=DefaultsConfig(llm=LLMConfig(model="ollama/llama3.2", temperature=0.5)),
            agents={
                "agent1": AgentConfig(description="d", prompt="p"),
            },
        )
        assert cfg.agents["agent1"].llm is not None
        assert cfg.agents["agent1"].llm.model == "ollama/llama3.2"
        assert cfg.agents["agent1"].llm.temperature == 0.5

    def test_explicit_llm_not_overwritten(self):
        cfg = AgentsConfig(
            defaults=DefaultsConfig(llm=LLMConfig(model="ollama/llama3.2")),
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    llm=LLMConfig(model="openai/gpt-4o", temperature=0.9),
                ),
            },
        )
        assert cfg.agents["agent1"].llm.model == "openai/gpt-4o"
        assert cfg.agents["agent1"].llm.temperature == 0.9

    def test_version_defaults_to_one(self):
        cfg = AgentsConfig()
        assert cfg.version == "1"

    def test_injectable_tools_empty_by_default(self):
        """No tools have inject_to_rag=True → injectable_tools is empty."""
        cfg = AgentsConfig(
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[ToolConfig(name="tool_a")],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == set()

    def test_injectable_tools_from_mcp(self):
        """MCP tools with inject_to_rag=True are collected."""
        cfg = AgentsConfig(
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[
                                ToolConfig(name="tool_a", inject_to_rag=True),
                                ToolConfig(name="tool_b"),
                            ],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"tool_a"}

    def test_injectable_tools_from_builtin(self):
        """Built-in tools with inject_to_rag=True are collected with builtin_ prefix."""
        cfg = AgentsConfig(
            tools={
                "format_date": BuiltinToolConfig(handler="m.f", inject_to_rag=True),
                "calc_rate": BuiltinToolConfig(handler="m.c"),
            },
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    tools=["format_date", "calc_rate"],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"builtin_format_date"}

    def test_injectable_tools_mixed(self):
        """Both MCP and built-in injectable tools are collected."""
        cfg = AgentsConfig(
            tools={
                "my_builtin": BuiltinToolConfig(handler="m.f", inject_to_rag=True),
            },
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    tools=["my_builtin"],
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[ToolConfig(name="mcp_tool", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tools == {"mcp_tool", "builtin_my_builtin"}

    def test_injectable_tool_ttls_empty_when_no_ttl(self):
        """inject_to_rag=True but rag_ttl=0 → no TTL entries (cache disabled)."""
        cfg = AgentsConfig(
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[ToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tool_ttls == {}

    def test_injectable_tool_ttls_from_agent_rag_ttl(self):
        """Agent-level rag_ttl propagates to all injectable tools without per-tool override."""
        cfg = AgentsConfig(
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    rag=RAGConfig(namespace="ns", rag_ttl=3600),
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[ToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].injectable_tool_ttls == {"tool_a": 3600}

    def test_injectable_tool_ttls_per_tool_override(self):
        """Per-tool rag_ttl overrides agent-level default."""
        cfg = AgentsConfig(
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    rag=RAGConfig(namespace="ns", rag_ttl=3600),
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[
                                ToolConfig(name="tool_a", inject_to_rag=True, rag_ttl=300),
                                ToolConfig(name="tool_b", inject_to_rag=True),
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
        from orchid_ai.config.schema import RAGDefaultsConfig

        cfg = AgentsConfig(
            defaults=DefaultsConfig(rag=RAGDefaultsConfig(rag_ttl=7200)),
            agents={
                "agent1": AgentConfig(
                    description="d",
                    prompt="p",
                    mcp_servers=[
                        MCPServerConfig(
                            name="mcp1",
                            url="http://x",
                            tools=[ToolConfig(name="tool_a", inject_to_rag=True)],
                        ),
                    ],
                ),
            },
        )
        assert cfg.agents["agent1"].rag.rag_ttl == 7200
        assert cfg.agents["agent1"].injectable_tool_ttls == {"tool_a": 7200}
