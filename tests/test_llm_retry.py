"""Tests for LLM automatic retry configuration and wiring (#5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import BaseChatModel

from orchid_ai.config.schema import (
    AgentConfig,
    AgentsConfig,
    DefaultsConfig,
    LLMConfig,
    RAGConfig,
)


# ── Schema tests ────────────────────────────────────────────


class TestLLMConfigRetry:
    """LLMConfig.retry_attempts field."""

    def test_default_no_retry(self):
        cfg = LLMConfig()
        assert cfg.retry_attempts == 0

    def test_retry_attempts_set(self):
        cfg = LLMConfig(model="gemini/gemini-2.5-flash", retry_attempts=3)
        assert cfg.retry_attempts == 3

    def test_retry_from_dict(self):
        data = {"model": "openai/gpt-4o", "retry_attempts": 5}
        cfg = LLMConfig(**data)
        assert cfg.retry_attempts == 5


class TestRetryInheritance:
    """Default retry propagates to agents unless overridden."""

    def test_agent_inherits_default_retry(self):
        config = AgentsConfig(
            defaults=DefaultsConfig(
                llm=LLMConfig(model="ollama/llama3.2", retry_attempts=3),
            ),
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        assert config.defaults.llm.retry_attempts == 3
        # Agent inherits default LLM (including retry)
        assert config.agents["test"].llm.retry_attempts == 3

    def test_agent_overrides_retry(self):
        config = AgentsConfig(
            defaults=DefaultsConfig(
                llm=LLMConfig(model="ollama/llama3.2", retry_attempts=3),
            ),
            agents={
                "custom": AgentConfig(
                    description="test",
                    prompt="test",
                    llm=LLMConfig(model="ollama/llama3.2", retry_attempts=5),
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        assert config.agents["custom"].llm.retry_attempts == 5


# ── Factory tests ───────────────────────────────────────────


class TestBuildChatModelRetry:
    """build_chat_model() with retry_attempts parameter."""

    def test_no_retry_returns_plain_model(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2", retry_attempts=0)
        assert isinstance(model, BaseChatModel)
        # Should NOT be a RunnableRetry
        class_name = type(model).__name__
        assert "Retry" not in class_name

    def test_with_retry_returns_retryable_model(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2", retry_attempts=3)
        # with_retry returns a RunnableRetry wrapper
        class_name = type(model).__name__
        assert "Retry" in class_name

    def test_retry_with_fallback(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model(
            "ollama/llama3.2",
            fallback_model="ollama/mistral",
            retry_attempts=3,
        )
        # Should have fallbacks AND each model should have retry
        assert hasattr(model, "fallbacks")
        assert len(model.fallbacks) == 1
        # The primary (runnable) should be a RunnableRetry
        assert "Retry" in type(model.runnable).__name__

    def test_retry_zero_same_as_disabled(self):
        from orchid_ai.llm_factory import build_chat_model

        model = build_chat_model("ollama/llama3.2", retry_attempts=0)
        class_name = type(model).__name__
        assert "Retry" not in class_name


# ── Graph wiring tests ──────────────────────────────────────


class TestGraphRetryWiring:
    """build_graph() creates models with correct retry configuration."""

    def test_default_retry_applied(self):
        """Graph builds successfully with default retry config."""
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = AgentsConfig(
            defaults=DefaultsConfig(
                llm=LLMConfig(model="ollama/llama3.2", retry_attempts=3),
            ),
            agents={
                "test": AgentConfig(
                    description="test agent",
                    prompt="You are a test agent",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None

    def test_agent_specific_retry(self):
        """Agent with its own retry gets a dedicated chat model."""
        from orchid_ai.graph.graph import _instantiate_agent

        agent_config = AgentConfig(
            description="critical agent",
            prompt="You are critical",
            llm=LLMConfig(model="ollama/llama3.2", retry_attempts=5),
            rag=RAGConfig(enabled=False),
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
            default_retry=0,  # agent wants 5, default is 0 → gets own model
        )

        # Agent should have its own chat model (not the default)
        assert agent._chat_model is not default_model

    def test_agent_inherits_default_when_retry_matches(self):
        """Agent with same retry as default reuses the shared model."""
        from orchid_ai.graph.graph import _instantiate_agent

        agent_config = AgentConfig(
            description="basic agent",
            prompt="You are basic",
            rag=RAGConfig(enabled=False),
            llm=LLMConfig(model="ollama/llama3.2", retry_attempts=3),
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
            default_retry=3,  # same as agent → reuses default
        )

        assert agent._chat_model is default_model

    def test_no_retry_no_crash(self):
        """Config without retry still works."""
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = AgentsConfig(
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                    llm=LLMConfig(model="ollama/llama3.2"),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None


# ── YAML round-trip test ────────────────────────────────────


class TestRetryYAML:
    """Verify retry_attempts survives YAML config loading."""

    def test_agents_yaml_with_retry(self):
        raw = {
            "defaults": {
                "llm": {
                    "model": "gemini/gemini-2.5-flash",
                    "retry_attempts": 3,
                },
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
                        "retry_attempts": 5,
                    },
                },
            },
        }
        config = AgentsConfig(**raw)

        assert config.defaults.llm.retry_attempts == 3
        assert config.agents["agent_a"].llm.retry_attempts == 3  # inherited
        assert config.agents["agent_b"].llm.retry_attempts == 5  # overridden

    def test_retry_with_fallback_yaml(self):
        """Both retry and fallback can coexist."""
        raw = {
            "defaults": {
                "llm": {
                    "model": "gemini/gemini-2.5-flash",
                    "fallback_model": "ollama/llama3.2",
                    "retry_attempts": 3,
                },
            },
            "agents": {
                "test": {
                    "description": "test",
                    "prompt": "test",
                },
            },
        }
        config = AgentsConfig(**raw)
        assert config.defaults.llm.fallback_model == "ollama/llama3.2"
        assert config.defaults.llm.retry_attempts == 3
