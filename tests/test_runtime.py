"""Tests for OrchidRuntime — typed dependency bag."""

from __future__ import annotations

from unittest.mock import MagicMock

from orchid_ai.core.mcp import OrchidMCPClient
from orchid_ai.core.repository import OrchidVectorReader
from orchid_ai.runtime import OrchidRuntime

# ── Defaults ────────────────────────────────────────────────────


class TestDefaults:
    def test_default_model(self):
        rt = OrchidRuntime()
        assert rt.default_model == ""

    def test_reader_defaults_to_none(self):
        rt = OrchidRuntime()
        assert rt.reader is None

    def test_get_reader_returns_null_when_none(self):
        from orchid_ai.rag.backends.null import NullVectorReader

        rt = OrchidRuntime()
        reader = rt.get_reader()
        assert isinstance(reader, NullVectorReader)

    def test_get_reader_returns_provided_reader(self):
        mock_reader = MagicMock(spec=OrchidVectorReader)
        rt = OrchidRuntime(reader=mock_reader)
        assert rt.get_reader() is mock_reader

    def test_get_chat_model_returns_default_when_none(self):
        import pytest

        rt = OrchidRuntime()
        with pytest.raises(RuntimeError, match="No LLM model configured"):
            rt.get_chat_model()

    def test_get_chat_model_returns_default_with_model(self):
        from langchain_core.language_models import BaseChatModel

        rt = OrchidRuntime(default_model="ollama/llama3.2")
        model = rt.get_chat_model()
        assert isinstance(model, BaseChatModel)

    def test_get_chat_model_returns_provided_model(self):
        from langchain_core.language_models import BaseChatModel

        mock_model = MagicMock(spec=BaseChatModel)
        rt = OrchidRuntime(chat_model=mock_model)
        assert rt.get_chat_model() is mock_model

    def test_get_mcp_client_factory_returns_default_callable(self):
        rt = OrchidRuntime()
        factory = rt.get_mcp_client_factory()
        assert callable(factory)

    def test_get_mcp_client_factory_returns_provided_factory(self):
        custom_factory = MagicMock()
        rt = OrchidRuntime(mcp_client_factory=custom_factory)
        assert rt.get_mcp_client_factory() is custom_factory


# ── Custom MCP factory ──────────────────────────────────────────


class TestCustomMCPFactory:
    def test_default_factory_creates_streamable_client(self):
        from orchid_ai.config.schema import OrchidMCPServerConfig
        from orchid_ai.mcp.client import StreamableHttpMCPClient

        rt = OrchidRuntime()
        factory = rt.get_mcp_client_factory()
        server_cfg = OrchidMCPServerConfig(name="test", url="http://localhost:8080")
        client = factory(server_cfg)
        assert isinstance(client, StreamableHttpMCPClient)
        assert client.server_url == "http://localhost:8080"

    def test_custom_factory_is_used(self):
        from orchid_ai.config.schema import OrchidMCPServerConfig

        mock_client = MagicMock(spec=OrchidMCPClient)
        custom_factory = MagicMock(return_value=mock_client)

        rt = OrchidRuntime(mcp_client_factory=custom_factory)
        factory = rt.get_mcp_client_factory()
        server_cfg = OrchidMCPServerConfig(name="test", url="http://x")
        result = factory(server_cfg)

        custom_factory.assert_called_once_with(server_cfg)
        assert result is mock_client


# ── Integration with build_graph ────────────────────────────────


class TestBuildGraphIntegration:
    def test_build_graph_accepts_runtime(self):
        """build_graph() should work with OrchidRuntime parameter."""
        from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidLLMConfig, OrchidRAGConfig
        from orchid_ai.graph.graph import build_graph

        config = OrchidAgentsConfig(
            agents={
                "test": OrchidAgentConfig(
                    description="test agent",
                    prompt="You are a test agent",
                    rag=OrchidRAGConfig(enabled=False),
                    llm=OrchidLLMConfig(model="test-model"),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="test-model")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None

    def test_build_graph_with_custom_reader(self):
        """build_graph() should use the reader from OrchidRuntime."""
        from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidLLMConfig, OrchidRAGConfig
        from orchid_ai.graph.graph import build_graph

        mock_reader = MagicMock(spec=OrchidVectorReader)
        runtime = OrchidRuntime(
            default_model="runtime-model",
            reader=mock_reader,
        )
        config = OrchidAgentsConfig(
            agents={
                "test": OrchidAgentConfig(
                    description="test agent",
                    prompt="You are a test agent",
                    rag=OrchidRAGConfig(enabled=False),
                    llm=OrchidLLMConfig(model="test-model"),
                ),
            },
        )
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None

    def test_build_graph_uses_custom_mcp_factory(self):
        """build_graph() should use the runtime's MCP factory for agent creation."""
        from orchid_ai.config.schema import (
            OrchidAgentConfig,
            OrchidAgentsConfig,
            OrchidLLMConfig,
            OrchidMCPServerConfig,
            OrchidRAGConfig,
        )
        from orchid_ai.graph.graph import build_graph

        mock_client = MagicMock(spec=OrchidMCPClient)
        custom_factory = MagicMock(return_value=mock_client)

        runtime = OrchidRuntime(
            default_model="test-model",
            mcp_client_factory=custom_factory,
        )
        config = OrchidAgentsConfig(
            agents={
                "test": OrchidAgentConfig(
                    description="test agent",
                    prompt="You are a test agent",
                    rag=OrchidRAGConfig(enabled=False),
                    llm=OrchidLLMConfig(model="test-model"),
                    mcp_servers=[OrchidMCPServerConfig(name="srv", url="http://x")],
                ),
            },
        )
        build_graph(config=config, runtime=runtime)
        # The factory should have been called for the MCP server config
        custom_factory.assert_called_once()


# ── SDK surface ─────────────────────────────────────────────────


class TestSDKSurface:
    def test_importable_from_orchid_ai(self):
        from orchid_ai import OrchidRuntime as Imported

        assert Imported is OrchidRuntime

    def test_in_all(self):
        import orchid_ai

        assert "OrchidRuntime" in orchid_ai.__all__
