"""
OrchidRuntime — typed dependency bag for framework consumers.

Holds the resolved dependencies needed by ``build_graph()`` and agent
instantiation.  Integrators construct one ``OrchidRuntime``, override
what they need, and pass it in — no need to understand internal wiring.

Example — using all defaults::

    runtime = OrchidRuntime(default_model="gemini/gemini-2.5-flash")
    graph = build_graph(config=config, runtime=runtime)

Example — custom chat model and MCP factory::

    runtime = OrchidRuntime(
        default_model="openai/gpt-4o",
        chat_model=ChatOpenAI(model="gpt-4o"),
        reader=my_qdrant_reader,
        mcp_client_factory=lambda cfg: MyMCPClient(cfg.url),
    )
    graph = build_graph(config=config, runtime=runtime)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from langchain_core.language_models import BaseChatModel

from .core.mcp import OrchidMCPClient, OrchidMCPClientRegistrationStore, OrchidMCPTokenStore
from .core.repository import OrchidVectorReader

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from .config.schema import OrchidMCPServerConfig
    from .mcp.auth_registry import OrchidMCPAuthRegistry

logger = logging.getLogger(__name__)

# Type alias for the MCP client factory callable.
# Takes a server config, returns a ready-to-use OrchidMCPClient.
MCPClientFactory = Callable[["OrchidMCPServerConfig"], OrchidMCPClient]


def _default_chat_model(model: str = "ollama/llama3.2", **kwargs) -> BaseChatModel:
    """Create the default LangChain chat model via the factory."""
    from .llm_factory import build_chat_model

    return build_chat_model(model, **kwargs)


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
    reader : OrchidVectorReader | None
        Vector store backend.  ``None`` → ``NullVectorReader`` (no RAG).
    chat_model : BaseChatModel | None
        LangChain chat model.  ``None`` → built via ``build_chat_model(default_model)``.
    mcp_client_factory : MCPClientFactory | None
        Factory for creating MCP clients from server config.
        ``None`` → ``StreamableHttpMCPClient`` (default).
    checkpointer : BaseCheckpointSaver | None
        LangGraph checkpointer for graph state persistence.
        ``None`` → no checkpointing (current request replays full history).
        When set, the compiled graph persists state keyed by ``thread_id``
        (= ``chat_id``).  Use :func:`orchid_ai.checkpointing.build_checkpointer`
        to create one from a type string or dotted class path.
    """

    default_model: str = "ollama/llama3.2"
    reader: OrchidVectorReader | None = None
    chat_model: BaseChatModel | None = None
    mcp_client_factory: MCPClientFactory | None = None
    mcp_token_store: OrchidMCPTokenStore | None = None
    mcp_client_registration_store: OrchidMCPClientRegistrationStore | None = None
    mcp_auth_registry: OrchidMCPAuthRegistry | None = field(default=None)
    checkpointer: BaseCheckpointSaver | None = None

    # ── Resolved accessors (lazy defaults) ──────────────────────

    def get_reader(self) -> OrchidVectorReader:
        """Return the configured reader, falling back to NullVectorReader."""
        if self.reader is not None:
            return self.reader
        from .rag.null import NullVectorReader

        return NullVectorReader()

    def get_chat_model(self) -> BaseChatModel:
        """Return the configured chat model, falling back to ChatLiteLLM."""
        if self.chat_model is not None:
            return self.chat_model
        return _default_chat_model(self.default_model)

    def get_mcp_client_factory(self) -> MCPClientFactory:
        """Return the configured MCP factory, falling back to the default.

        When no explicit ``mcp_client_factory`` was supplied, returns a
        callable bound to this runtime's token + client-registration
        stores so ``oauth`` servers can resolve per-user tokens and
        refresh them against the discovered token endpoint.
        """
        if self.mcp_client_factory is not None:
            return self.mcp_client_factory
        token_store = self.mcp_token_store
        registration_store = self.mcp_client_registration_store
        return lambda cfg: self.default_mcp_client_factory(
            cfg,
            token_store=token_store,
            registration_store=registration_store,
        )

    # ── Default MCP factory (override in subclasses) ────────────

    @staticmethod
    def default_mcp_client_factory(
        server_config: OrchidMCPServerConfig,
        *,
        token_store: OrchidMCPTokenStore | None = None,
        registration_store: OrchidMCPClientRegistrationStore | None = None,
    ) -> OrchidMCPClient:
        """Create a ``StreamableHttpMCPClient`` from the server config.

        Override in a subclass to change the default transport / client
        without having to supply a full ``mcp_client_factory`` callable.

        Auth modes (unchanged contract):
          - ``none`` (default): no auth headers sent.
          - ``passthrough``: forwards the graph ``OrchidAuthContext`` bearer token.
          - ``oauth``: resolves per-user tokens from *token_store* and
            refreshes against the discovered endpoint persisted in
            *registration_store* (populated by the API's auth router on
            first Connect click).
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
            registration_store=registration_store,
        )
