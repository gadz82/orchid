"""Tests for LLM response caching configuration (#4)."""

from __future__ import annotations

from unittest.mock import patch

from orchid_ai.config.schema import (
    AgentConfig,
    AgentsConfig,
    DefaultsConfig,
    LLMConfig,
    RAGConfig,
)


# ── Schema tests ────────────────────────────────────────────


class TestCacheConfig:
    """DefaultsConfig.cache_enabled field."""

    def test_default_cache_disabled(self):
        cfg = DefaultsConfig()
        assert cfg.cache_enabled is False

    def test_cache_enabled(self):
        cfg = DefaultsConfig(cache_enabled=True)
        assert cfg.cache_enabled is True

    def test_cache_from_yaml_dict(self):
        raw = {
            "cache_enabled": True,
            "llm": {"model": "ollama/llama3.2"},
        }
        cfg = DefaultsConfig(**raw)
        assert cfg.cache_enabled is True


class TestCacheInAgentsConfig:
    """cache_enabled in full AgentsConfig."""

    def test_cache_enabled_in_full_config(self):
        raw = {
            "defaults": {
                "cache_enabled": True,
                "llm": {"model": "ollama/llama3.2"},
            },
            "agents": {
                "test": {
                    "description": "test",
                    "prompt": "test",
                },
            },
        }
        config = AgentsConfig(**raw)
        assert config.defaults.cache_enabled is True

    def test_cache_disabled_by_default_in_full_config(self):
        config = AgentsConfig(
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        assert config.defaults.cache_enabled is False


# ── Graph wiring tests ──────────────────────────────────────


class TestCacheGraphWiring:
    """build_graph() sets up LLM cache when enabled."""

    def test_cache_enabled_calls_set_llm_cache(self):
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = AgentsConfig(
            defaults=DefaultsConfig(
                cache_enabled=True,
                llm=LLMConfig(model="ollama/llama3.2"),
            ),
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")

        with patch("langchain_core.globals.set_llm_cache") as mock_set:
            build_graph(config=config, runtime=runtime)
            mock_set.assert_called_once()

    def test_cache_disabled_no_set_llm_cache(self):
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = AgentsConfig(
            defaults=DefaultsConfig(
                cache_enabled=False,
                llm=LLMConfig(model="ollama/llama3.2"),
            ),
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")

        with patch("langchain_core.globals.set_llm_cache") as mock_set:
            build_graph(config=config, runtime=runtime)
            mock_set.assert_not_called()

    def test_graph_compiles_with_cache_enabled(self):
        from orchid_ai.graph.graph import build_graph
        from orchid_ai.runtime import OrchidRuntime

        config = AgentsConfig(
            defaults=DefaultsConfig(
                cache_enabled=True,
                llm=LLMConfig(model="ollama/llama3.2"),
            ),
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None
