"""
OrchidRuntime — typed dependency bag for framework consumers.

Holds the resolved dependencies needed by ``build_graph()`` and agent
instantiation.  Integrators construct one ``OrchidRuntime``, override
what they need, and pass it in — no need to understand internal wiring.

Example — using all defaults::

    runtime = OrchidRuntime(default_model="gemini/gemini-2.5-flash")
    graph = build_graph(config=config, runtime=runtime)

Example — custom LLM provider and MCP factory::

    runtime = OrchidRuntime(
        default_model="openai/gpt-4o",
        llm_service=MyCustomLLMProvider(),
        reader=my_qdrant_reader,
        mcp_client_factory=lambda cfg: MyMCPClient(cfg.url),
    )
    graph = build_graph(config=config, runtime=runtime)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .core.llm_provider import LLMProvider
from .core.mcp import MCPClient, MCPTokenStore
from .core.repository import VectorReader

if TYPE_CHECKING:
    from .config.schema import MCPServerConfig
    from .mcp.auth_registry import MCPAuthRegistry

logger = logging.getLogger(__name__)

# Type alias for the MCP client factory callable.
# Takes a server config, returns a ready-to-use MCPClient.
MCPClientFactory = Callable[["MCPServerConfig"], MCPClient]


def _default_mcp_client_factory(
    server_config: MCPServerConfig,
    *,
    token_store: MCPTokenStore | None = None,
) -> MCPClient:
    """Create a StreamableHttpMCPClient from server config (default factory).

    Auth modes:
      - ``none`` (default): no auth headers sent.
      - ``passthrough``: forwards graph AuthContext bearer token.
      - ``oauth``: resolves per-user tokens from the *token_store*.
    """
    from .mcp.client import StreamableHttpMCPClient

    return StreamableHttpMCPClient(
        server_config.url,
        server_type=server_config.type,
        transport=server_config.transport,
        cache_ttl=server_config.cache_ttl,
        server_name=server_config.name,
        auth_mode=server_config.auth.mode,
        token_store=token_store,
        token_endpoint=server_config.auth.token_endpoint,
        client_id=server_config.auth.client_id,
    )


def _default_llm_service() -> LLMProvider:
    """Create the default LiteLLM-backed provider."""
    from .llm_service import LiteLLMProvider

    return LiteLLMProvider()


@dataclass
class OrchidRuntime:
    """
    Resolved dependencies for the Orchid framework.

    Integrators construct this once and pass it to ``build_graph()``.
    Every field has a sensible default — override only what you need.

    Attributes
    ----------
    default_model : str
        LiteLLM model identifier (e.g. ``"gemini/gemini-2.5-flash"``).
    reader : VectorReader | None
        Vector store backend.  ``None`` → ``NullVectorReader`` (no RAG).
    llm_service : LLMProvider | None
        LLM abstraction.  ``None`` → ``LiteLLMProvider()`` (default).
    mcp_client_factory : MCPClientFactory | None
        Factory for creating MCP clients from server config.
        ``None`` → ``StreamableHttpMCPClient`` (default).
    """

    default_model: str = "ollama/llama3.2"
    reader: VectorReader | None = None
    llm_service: LLMProvider | None = None
    mcp_client_factory: MCPClientFactory | None = None
    mcp_token_store: MCPTokenStore | None = None
    mcp_auth_registry: MCPAuthRegistry | None = field(default=None)

    # ── Resolved accessors (lazy defaults) ──────────────────────

    def get_reader(self) -> VectorReader:
        """Return the configured reader, falling back to NullVectorReader."""
        if self.reader is not None:
            return self.reader
        from .rag.null import NullVectorReader

        return NullVectorReader()

    def get_llm_service(self) -> LLMProvider:
        """Return the configured LLM provider, falling back to LiteLLMProvider."""
        if self.llm_service is not None:
            return self.llm_service
        return _default_llm_service()

    def get_mcp_client_factory(self) -> MCPClientFactory:
        """Return the configured MCP factory, falling back to StreamableHttpMCPClient."""
        if self.mcp_client_factory is not None:
            return self.mcp_client_factory
        return _default_mcp_client_factory
