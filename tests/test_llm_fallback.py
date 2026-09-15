"""Tests for LLM fallback model configuration and wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import BaseChatModel

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    OrchidLLMConfig,
    OrchidRAGConfig,
    OrchidSupervisorConfig,
)

# ── Schema tests ────────────────────────────────────────────


class TestLLMConfigFallback:
    """OrchidLLMConfig.fallback_model field."""

    def test_default_no_fallback(self):
        cfg = OrchidLLMConfig()
        assert cfg.fallback_model is None

    def test_fallback_model_set(self):
        cfg = OrchidLLMConfig(model="gemini/gemini-2.5-flash", fallback_model="ollama/llama3.2")
        assert cfg.model == "gemini/gemini-2.5-flash"
        assert cfg.fallback_model == "ollama/llama3.2"

    def test_fallback_model_from_dict(self):
        """Parse from dict (as YAML loader produces)."""
        data = {"model": "openai/gpt-4o", "fallback_model": "anthropic/claude-sonnet-4-20250514"}
        cfg = OrchidLLMConfig(**data)
        assert cfg.fallback_model == "anthropic/claude-sonnet-4-20250514"

    def test_supervisor_fallback(self):
        cfg = OrchidSupervisorConfig(fallback_model="ollama/llama3.2")
        assert cfg.fallback_model == "ollama/llama3.2"

    def test_supervisor_default_no_fallback(self):
        cfg = OrchidSupervisorConfig()
        assert cfg.fallback_model is None


class TestFallbackInheritance:
    """Default fallback propagates to agents unless overridden."""

    def test_default_fallback_in_config(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                llm=OrchidLLMConfig(model="gemini/gemini-2.5-flash", fallback_model="ollama/llama3.2"),
            ),
            agents={
                "test": OrchidAgentConfig(description="test", prompt="test", rag=OrchidRAGConfig(enabled=False)),
            },
        )
        # Default fallback is set
        assert config.defaults.llm.fallback_model == "ollama/llama3.2"
        # Agent inherits default LLM (including fallback)
        assert config.agents["test"].llm is not None
        assert config.agents["test"].llm.model == "gemini/gemini-2.5-flash"

    def test_agent_overrides_fallback(self):
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                llm=OrchidLLMConfig(model="gemini/gemini-2.5-flash", fallback_model="ollama/llama3.2"),
            ),
            agents={
                "custom": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    llm=OrchidLLMConfig(model="openai/gpt-4o", fallback_model="anthropic/claude-sonnet-4-20250514"),
                    rag=OrchidRAGConfig(enabled=False),
                ),
            },
        )
        assert config.agents["custom"].llm.model == "openai/gpt-4o"
        assert config.agents["custom"].llm.fallback_model == "anthropic/claude-sonnet-4-20250514"

    def test_agent_with_custom_model_keeps_own_fallback(self):
        """Agent with custom model and explicit fallback_model keeps its own."""
        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                llm=OrchidLLMConfig(model="gemini/gemini-2.5-flash", fallback_model="ollama/llama3.2"),
            ),
            agents={
                "custom_fb": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    llm=OrchidLLMConfig(model="openai/gpt-4o", fallback_model="anthropic/claude-sonnet-4-20250514"),
                    rag=OrchidRAGConfig(enabled=False),
                ),
                "no_custom_llm": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    rag=OrchidRAGConfig(enabled=False),
                ),
            },
        )
        # Agent with explicit LLM keeps its own fallback
        assert config.agents["custom_fb"].llm.fallback_model == "anthropic/claude-sonnet-4-20250514"
        # Agent without explicit LLM inherits default (including fallback)
        assert config.agents["no_custom_llm"].llm.fallback_model == "ollama/llama3.2"


# ── Factory tests ───────────────────────────────────────────


class TestBuildChatModelFallback:
    """build_chat_model() with fallback parameter."""

    def test_no_fallback_returns_single_model(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2")
        assert isinstance(model, BaseChatModel)
        # Should NOT be a RunnableWithFallbacks
        assert not hasattr(model, "fallbacks")

    def test_with_fallback_returns_model_with_fallbacks(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2", fallback_model="ollama/mistral")
        # with_fallbacks returns a RunnableWithFallbacks which has a .fallbacks attribute
        assert hasattr(model, "fallbacks")
        assert len(model.fallbacks) == 1

    def test_fallback_none_same_as_no_fallback(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2", fallback_model=None)
        assert isinstance(model, BaseChatModel)
        assert not hasattr(model, "fallbacks")


# ── Graph wiring tests ──────────────────────────────────────


class TestGraphFallbackWiring:
    """build_graph() creates per-agent models with correct fallbacks."""

    def test_default_fallback_applied_to_agents(self):
        """Agents without custom LLM config get the default fallback."""
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                llm=OrchidLLMConfig(model="ollama/llama3.2", fallback_model="ollama/mistral"),
            ),
            agents={
                "test": OrchidAgentConfig(
                    description="test agent",
                    prompt="You are a test agent",
                    rag=OrchidRAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None

    def test_agent_specific_fallback(self):
        """Agent with its own fallback gets a dedicated chat model."""
        from orchid_ai.graph.graph import _instantiate_agent

        agent_config = OrchidAgentConfig(
            description="critical agent",
            prompt="You are critical",
            llm=OrchidLLMConfig(model="openai/gpt-4o", fallback_model="anthropic/claude-sonnet-4-20250514"),
            rag=OrchidRAGConfig(enabled=False),
        )
        agent_config.name = "critical"

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(return_value=[])

        default_model = MagicMock(spec=BaseChatModel)

        agent = _instantiate_agent(
            "critical",
            agent_config,
            "ollama/llama3.2",
            mock_reader,
            default_chat_model=default_model,
            default_fallback=None,
        )

        # Agent should have its own chat model (not the default)
        assert agent._chat_model is not default_model
        # The chat model should have fallbacks
        assert hasattr(agent._chat_model, "fallbacks")

    def test_agent_inherits_default_when_no_custom_llm(self):
        """Agent without custom LLM uses the shared default chat model."""
        from orchid_ai.graph.graph import _instantiate_agent

        agent_config = OrchidAgentConfig(
            description="basic agent",
            prompt="You are basic",
            rag=OrchidRAGConfig(enabled=False),
            llm=OrchidLLMConfig(model="ollama/llama3.2"),
        )
        agent_config.name = "basic"

        mock_reader = MagicMock()
        mock_reader.retrieve = AsyncMock(return_value=[])

        default_model = MagicMock(spec=BaseChatModel)

        agent = _instantiate_agent(
            "basic",
            agent_config,
            "ollama/llama3.2",
            mock_reader,
            default_chat_model=default_model,
            default_fallback=None,
        )

        # Agent should reuse the shared default chat model
        assert agent._chat_model is default_model

    def test_supervisor_fallback_override(self):
        """Supervisor with its own fallback gets a dedicated chat model."""
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(llm=OrchidLLMConfig(model="ollama/llama3.2")),
            supervisor=OrchidSupervisorConfig(fallback_model="ollama/mistral"),
            agents={
                "test": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    rag=OrchidRAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None

    def test_no_fallback_no_crash(self):
        """Config without any fallback_model still works."""
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = OrchidAgentsConfig(
            agents={
                "test": OrchidAgentConfig(
                    description="test",
                    prompt="test",
                    rag=OrchidRAGConfig(enabled=False),
                    llm=OrchidLLMConfig(model="ollama/llama3.2"),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None


# ── YAML round-trip test ────────────────────────────────────


class TestFallbackYAML:
    """Verify fallback_model survives YAML config loading."""

    def test_agents_yaml_with_fallback(self):
        """OrchidAgentsConfig parses fallback_model from dict (simulates YAML)."""
        raw = {
            "defaults": {
                "llm": {
                    "model": "gemini/gemini-2.5-flash",
                    "fallback_model": "ollama/llama3.2",
                },
            },
            "supervisor": {
                "assistant_name": "Test AI",
                "fallback_model": "ollama/mistral",
            },
            "agents": {
                "agent_a": {
                    "description": "Agent A",
                    "prompt": "You are agent A",
                },
                "agent_b": {
                    "description": "Agent B",
                    "prompt": "You are agent B",
                    "llm": {
                        "model": "openai/gpt-4o",
                        "fallback_model": "anthropic/claude-sonnet-4-20250514",
                    },
                },
            },
        }
        config = OrchidAgentsConfig(**raw)

        assert config.defaults.llm.fallback_model == "ollama/llama3.2"
        assert config.supervisor.fallback_model == "ollama/mistral"
        assert config.agents["agent_a"].llm.fallback_model == "ollama/llama3.2"  # inherited from defaults
        assert config.agents["agent_b"].llm.fallback_model == "anthropic/claude-sonnet-4-20250514"
