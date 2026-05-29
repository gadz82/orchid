"""
Orchid — platform-agnostic multi-agent AI framework.

Public SDK surface — import the canonical entry point plus common types::

    from orchid_ai import Orchid, OrchidAgent, OrchidAuthContext, load_config

:class:`Orchid` is the **mandatory** integrator bootstrap class. The
library does NOT export a free ``build_runtime`` function anymore —
``orchid-api``, ``orchid-cli``, and in-process integrators all route
through :class:`Orchid` so the three surfaces stay in lock-step.
"""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("orchid-ai")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

from .checkpointing.factory import build_checkpointer, shutdown_checkpointer
from .config.frontmatter import MarkdownFile, load_markdown_file, parse_frontmatter
from .config.loader import load_config
from .config.errors import OrchidConfigError
from .config.md_loader import load_md_config, md_infrastructure_to_env
from .config.watcher import (
    ConfigSnapshot,
    OrchidConfigSnapshot,
    OrchidConfigWatcher,
    OrchidConfigWatcherBase,
    OrchidYamlConfigWatcher,
    YamlConfigWatcher,
)
from .core.agent import OrchidAgent
from .core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
    OrchidGuardrailResult,
)
from .core.auth_config import (
    OrchidAuthConfigProvider,
    OrchidAuthExchangeClient,
    OrchidAuthExchangeError,
    OrchidUpstreamOAuthConfig,
    OrchidUpstreamTokenResponse,
)
from .core.doc_store import OrchidDocStore
from .core.graph_store import (
    OrchidEdge,
    OrchidEntity,
    OrchidEntityExtractor,
    OrchidGraphStore,
)
from .core.identity import OrchidIdentityError, OrchidIdentityResolver
from .core.ingestion import OrchidChunk, OrchidChunkPostProcessor, OrchidIngestionStrategy
from .core.mcp import (
    OrchidMCPAuthRequiredError,
    OrchidMCPClient,
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPDiscoverable,
    OrchidMCPDiscoveryError,
    OrchidMCPTokenRecord,
    OrchidMCPTokenStore,
    OrchidMCPToolCaller,
    OrchidMCPToolResult,
)
from .core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayToken,
    OrchidMCPGatewayTokenStore,
)
from .core.repository import (
    Document,
    OrchidDocument,
    OrchidSearchResult,
    OrchidVectorReader,
    OrchidVectorStoreAdmin,
    OrchidVectorStoreRepository,
    OrchidVectorWriter,
)
from .core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from .core.sparse import OrchidSparseEncoder, OrchidSparseVector
from .core.state import OrchidAgentState, OrchidAuthContext
from .core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput
from .graph.graph import build_graph
from .graph.supervisor import OrchidRoutingDecision
from .guardrails import build_guardrail_chain, register_guardrail
from .llm_factory import build_chat_model
from .mcp.auth_registry import OrchidMCPAuthRegistry, OrchidMCPOAuthServerInfo
from .mcp.discovery import (
    OrchidMCPAuthDiscovery,
    extract_resource_metadata_url,
    probe_mcp_server_for_resource_metadata,
)
from .mcp.inventory import OrchidMCPServerEntry, OrchidMCPServerInventory
from .mcp.oauth_state import (
    OrchidInMemoryOAuthStateStore,
    OrchidOAuthPendingState,
    OrchidOAuthStateStore,
    build_oauth_state_store,
    register_oauth_state_store,
)
from .mcp.session_warmer import OrchidSessionWarmer, OrchidWarmReport
from .observability.callbacks import OrchidMetricsHandler
from .orchid import Orchid, OrchidInvokeResult, OrchidPendingApproval
from .persistence.base import OrchidChatStorage
from .persistence.factory import build_chat_storage
from .config.storage import OrchidConfigStorage
from .config.storage_factory import build_config_storage
from .config.schema_storage import OrchidConfigStorageConfig
from .persistence.mcp_client_registration_factory import build_mcp_client_registration_store
from .persistence.mcp_client_registration_sqlite import OrchidSQLiteMCPClientRegistrationStore
from .persistence.mcp_token_factory import build_mcp_token_store
from .persistence.mcp_token_sqlite import OrchidSQLiteMCPTokenStore
from .persistence.config_sqlite import OrchidSQLiteConfigStorage
from .persistence.sqlite import OrchidSQLiteChatStorage
from .plugins import iter_entry_point_plugins
from .documents.post_processors import (
    ContextualHeaderPostProcessor,
    EntityExtractionPostProcessor,
    LLMEntityExtractor,
    OrchidExtractedGraph,
)
from .documents.strategies import (
    HeaderedIngestion,
    HierarchicalIngestion,
    RecursiveIngestion,
    SemanticIngestion,
    get_ingestion_strategy,
    get_post_processor,
    register_ingestion_strategy,
    register_post_processor,
)
from .rag.factory import (
    build_doc_store,
    build_graph_store,
    build_reader,
    build_sparse_encoder,
    register_doc_store_backend,
    register_graph_store_backend,
    register_sparse_encoder_backend,
    register_vector_backend,
)
from .rag.adapters import (
    from_langchain_document,
    from_langchain_documents,
    to_langchain_document,
    to_langchain_documents,
)
from .rag.scopes import OrchidRAGScope
from .rag.strategies import (
    GraphRAGRetrieval,
    HybridRetrieval,
    HyDERetrieval,
    MultiQueryRetrieval,
    SimpleRetrieval,
    get_retrieval_strategy,
    register_retrieval_strategy,
)
from .rag.transformers import (
    DecomposeTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    ReformulateTransformer,
    get_query_transformer,
    register_query_transformer,
)
from .runtime import OrchidRuntime
from .tools import FunctionTool, OrchidToolRegistry

# ``utils.import_class`` is intentionally NOT re-exported on the top-level
# ``orchid_ai`` namespace — it's an implementation detail used by the
# framework's own factories (chat storage, MCP store, checkpointers,
# identity resolver) and can change without notice.  Integrators who
# genuinely need dynamic dotted-path resolution should import it
# explicitly from :mod:`orchid_ai.utils`.

__all__ = [
    # ── Mandatory entry point ─────────────────────────
    "Orchid",
    "OrchidInvokeResult",
    "OrchidPendingApproval",
    # ── Core ABCs (integrator subclass targets) ───────
    "OrchidAgent",
    "OrchidAgentState",
    "OrchidAuthConfigProvider",
    "OrchidAuthContext",
    "OrchidAuthExchangeClient",
    "OrchidAuthExchangeError",
    "OrchidChatStorage",
    "OrchidConfigError",
    "OrchidChunk",
    "OrchidChunkPostProcessor",
    "OrchidDocStore",
    "OrchidEdge",
    "OrchidEntity",
    "OrchidEntityExtractor",
    "OrchidGraphStore",
    "OrchidIdentityError",
    "OrchidIdentityResolver",
    "OrchidIngestionStrategy",
    "OrchidMCPClient",
    "OrchidMCPClientRegistrationStore",
    "OrchidMCPDiscoverable",
    "OrchidMCPGatewayAuthCodeStore",
    "OrchidMCPGatewayClientStore",
    "OrchidMCPGatewayTokenStore",
    "OrchidMCPToolCaller",
    "OrchidMCPTokenStore",
    "OrchidQueryTransformer",
    "OrchidRetrievalStrategy",
    "OrchidSparseEncoder",
    "OrchidSparseVector",
    "OrchidTool",
    "OrchidToolInput",
    "OrchidToolOutput",
    "OrchidToolRegistry",
    "OrchidVectorReader",
    "OrchidVectorStoreAdmin",
    "OrchidVectorStoreRepository",
    "OrchidVectorWriter",
    # ── Data / result types ───────────────────────────
    "Document",
    "OrchidDocument",
    "OrchidMCPAuthDiscovery",
    "OrchidMCPAuthRegistry",
    "OrchidMCPAuthRequiredError",
    "OrchidMCPClientRegistration",
    "OrchidMCPDiscoveryError",
    "OrchidMCPGatewayAuthCode",
    "OrchidMCPGatewayClient",
    "OrchidMCPGatewayToken",
    "OrchidMCPOAuthServerInfo",
    "OrchidMCPServerEntry",
    "OrchidMCPServerInventory",
    "OrchidSessionWarmer",
    "OrchidWarmReport",
    "extract_resource_metadata_url",
    "probe_mcp_server_for_resource_metadata",
    "OrchidMCPTokenRecord",
    "OrchidMCPToolResult",
    "OrchidRAGScope",
    "OrchidRoutingDecision",
    "OrchidSearchResult",
    "OrchidUpstreamOAuthConfig",
    "OrchidUpstreamTokenResponse",
    # ── MCP OAuth state store ─────────────────────────
    "OrchidInMemoryOAuthStateStore",
    "OrchidOAuthPendingState",
    "OrchidOAuthStateStore",
    "build_oauth_state_store",
    "register_oauth_state_store",
    # ── Guardrails ────────────────────────────────────
    "OrchidGuardrail",
    "OrchidGuardrailAction",
    "OrchidGuardrailChain",
    "OrchidGuardrailContext",
    "OrchidGuardrailDirection",
    "OrchidGuardrailResult",
    "build_guardrail_chain",
    "register_guardrail",
    # ── Built-in backends ─────────────────────────────
    "OrchidConfigStorage",
    "OrchidConfigStorageConfig",
    "OrchidSQLiteChatStorage",
    "OrchidSQLiteConfigStorage",
    "OrchidSQLiteMCPClientRegistrationStore",
    "OrchidSQLiteMCPTokenStore",
    # ── Runtime + observability ───────────────────────
    "OrchidRuntime",
    "OrchidMetricsHandler",
    # ── Built-in ingestion strategies + post-processors ─
    "ContextualHeaderPostProcessor",
    "EntityExtractionPostProcessor",
    "HeaderedIngestion",
    "HierarchicalIngestion",
    "LLMEntityExtractor",
    "OrchidExtractedGraph",
    "RecursiveIngestion",
    "SemanticIngestion",
    # ── Built-in retrieval strategies + transformers ──
    "DecomposeTransformer",
    "GraphRAGRetrieval",
    "HybridRetrieval",
    "HyDERetrieval",
    "HyDETransformer",
    "MultiQueryRetrieval",
    "MultiQueryTransformer",
    "ReformulateTransformer",
    "SimpleRetrieval",
    # ── Factories + RAG registries ────────────────────
    "ConfigSnapshot",
    "FunctionTool",
    "OrchidConfigSnapshot",
    "OrchidConfigWatcher",
    "OrchidConfigWatcherBase",
    "OrchidYamlConfigWatcher",
    "YamlConfigWatcher",
    "build_chat_model",
    "build_chat_storage",
    "build_checkpointer",
    "build_config_storage",
    "build_doc_store",
    "build_graph",
    "build_graph_store",
    "build_mcp_client_registration_store",
    "build_mcp_token_store",
    "build_reader",
    "build_sparse_encoder",
    "from_langchain_document",
    "from_langchain_documents",
    "get_ingestion_strategy",
    "get_post_processor",
    "get_query_transformer",
    "get_retrieval_strategy",
    "iter_entry_point_plugins",
    "load_config",
    "load_markdown_file",
    "load_md_config",
    "MarkdownFile",
    "md_infrastructure_to_env",
    "parse_frontmatter",
    "register_doc_store_backend",
    "register_graph_store_backend",
    "register_ingestion_strategy",
    "register_post_processor",
    "register_query_transformer",
    "register_retrieval_strategy",
    "register_sparse_encoder_backend",
    "register_vector_backend",
    "shutdown_checkpointer",
    "to_langchain_document",
    "to_langchain_documents",
]
