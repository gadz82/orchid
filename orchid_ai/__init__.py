"""
Orchid — platform-agnostic multi-agent AI framework.

The public API has two tiers.

**RUN** — drive the framework through the single facade :class:`Orchid`::

    from orchid_ai import Orchid

    async with Orchid.from_config_path("orchid.yml") as orchid:
        result = await orchid.invoke("Hello", user_id="alice", tenant_id="acme")
        print(result.response)

:class:`Orchid` is the one bootstrap/runtime entry point — ``orchid-api``,
``orchid-cli`` and in-process integrators all route through it (there is no
free ``build_runtime``). It owns construction (:meth:`Orchid.from_config_path`
/ :meth:`Orchid.from_md_config`), execution (``invoke`` / ``stream`` /
``resume``), lifecycle (``close`` / ``reload_config`` / ``async with``) and
read-only accessors (``graph`` / ``runtime`` / ``config`` / ``chat_repo`` / …).

**EXTEND** — subclass the ABCs and register implementations exported below
(:class:`OrchidAgent`, :class:`OrchidIdentityResolver`,
:class:`OrchidChatStorage`, :class:`OrchidVectorReader`, the guardrail /
ingestion / retrieval contracts, and the ``register_*`` hooks). These are
*import-time* extension points: you subclass or register them **before** an
``Orchid`` instance exists, so they live alongside the facade rather than
behind it.

Everything else is lower-level plumbing and is intentionally **not**
re-exported here — import it from its submodule when you genuinely need it::

    from orchid_ai.checkpointing import build_checkpointer
    from orchid_ai.rag.factory import build_reader
    from orchid_ai.persistence.sqlite import OrchidSQLiteChatStorage
    from orchid_ai.observability import OrchidMetricsHandler
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("orchid-ai")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

# ── RUN — the single facade ──────────────────────────────────
from .orchid import Orchid, OrchidInvokeResult, OrchidPendingApproval

# ── CONFIGURE / BOOTSTRAP ────────────────────────────────────
from .config.errors import OrchidConfigError
from .config.loader import load_config
from .runtime import OrchidRuntime

# ── EXTEND — agents ──────────────────────────────────────────
from .agents.generic_agent import GenericAgent
from .core.agent import OrchidAgent
from .core.run_config import CONFIG_KEY_AUTH, auth_from_config, with_auth
from .core.state import OrchidAgentState, OrchidAuthContext

# ── EXTEND — identity & auth ─────────────────────────────────
from .core.auth_config import (
    OrchidAuthConfigProvider,
    OrchidAuthExchangeClient,
    OrchidAuthExchangeError,
    OrchidUpstreamOAuthConfig,
    OrchidUpstreamTokenResponse,
)
from .core.identity import OrchidIdentityError, OrchidIdentityResolver

# ── EXTEND — storage ─────────────────────────────────────────
from .config.storage import OrchidConfigStorage
from .persistence.base import OrchidChatStorage

# ── EXTEND — vector / RAG / documents ────────────────────────
from .core.doc_store import OrchidDocStore
from .core.graph_store import OrchidEdge, OrchidEntity, OrchidEntityExtractor, OrchidGraphStore
from .core.ingestion import OrchidChunk, OrchidChunkPostProcessor, OrchidIngestionStrategy
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
from .rag.scopes import OrchidRAGScope

# ── EXTEND — MCP ─────────────────────────────────────────────
from .core.mcp import (
    OrchidMCPAuthRequiredError,
    OrchidMCPClient,
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPDiscoverable,
    OrchidMCPTokenRecord,
    OrchidMCPTokenStore,
    OrchidMCPToolCaller,
    OrchidMCPToolResult,
)
from .core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayTokenStore,
)
from .mcp.oauth_state import OrchidOAuthStateStore, register_oauth_state_store

# ── EXTEND — tools ───────────────────────────────────────────
from .core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput

# ── EXTEND — guardrails ──────────────────────────────────────
from .core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
    OrchidGuardrailResult,
)

# ── EXTEND — content ─────────────────────────────────────────
from .core.content import OrchidContentItem, OrchidContentSource

# ── EXTEND — registries (register implementations at import time) ──
from .guardrails import register_guardrail
from .documents.strategies import register_ingestion_strategy, register_post_processor
from .rag.factory import (
    register_doc_store_backend,
    register_graph_store_backend,
    register_sparse_encoder_backend,
    register_vector_backend,
)
from .rag.strategies import register_retrieval_strategy
from .rag.transformers import register_query_transformer

# ``utils.import_class`` and the ``build_*`` / ``get_*`` factories,
# LangChain adapters, concrete built-in backends/strategies, config
# watchers and observability handlers are deliberately NOT re-exported
# here.  They are implementation/plumbing details reachable from their
# own submodules (see the module docstring).  This keeps ``Orchid`` and
# the extension ABCs the obvious top-level surface.

__all__ = [
    "__version__",
    # ── RUN — the single facade ───────────────────────
    "Orchid",
    "OrchidInvokeResult",
    "OrchidPendingApproval",
    # ── Configure / bootstrap ─────────────────────────
    "load_config",
    "OrchidConfigError",
    "OrchidRuntime",
    # ── EXTEND — agents ───────────────────────────────
    "OrchidAgent",
    "GenericAgent",
    "OrchidAgentState",
    "OrchidAuthContext",
    "CONFIG_KEY_AUTH",
    "auth_from_config",
    "with_auth",
    # ── EXTEND — identity & auth ──────────────────────
    "OrchidIdentityResolver",
    "OrchidIdentityError",
    "OrchidAuthConfigProvider",
    "OrchidAuthExchangeClient",
    "OrchidAuthExchangeError",
    "OrchidUpstreamOAuthConfig",
    "OrchidUpstreamTokenResponse",
    # ── EXTEND — storage ──────────────────────────────
    "OrchidChatStorage",
    "OrchidConfigStorage",
    # ── EXTEND — vector / RAG / documents ─────────────
    "Document",
    "OrchidDocument",
    "OrchidSearchResult",
    "OrchidVectorReader",
    "OrchidVectorWriter",
    "OrchidVectorStoreAdmin",
    "OrchidVectorStoreRepository",
    "OrchidDocStore",
    "OrchidGraphStore",
    "OrchidEntity",
    "OrchidEdge",
    "OrchidEntityExtractor",
    "OrchidIngestionStrategy",
    "OrchidChunk",
    "OrchidChunkPostProcessor",
    "OrchidRetrievalStrategy",
    "OrchidQueryTransformer",
    "OrchidSparseEncoder",
    "OrchidSparseVector",
    "OrchidRAGScope",
    # ── EXTEND — MCP ──────────────────────────────────
    "OrchidMCPClient",
    "OrchidMCPToolCaller",
    "OrchidMCPDiscoverable",
    "OrchidMCPToolResult",
    "OrchidMCPTokenStore",
    "OrchidMCPTokenRecord",
    "OrchidMCPClientRegistrationStore",
    "OrchidMCPClientRegistration",
    "OrchidMCPAuthRequiredError",
    "OrchidMCPGatewayAuthCodeStore",
    "OrchidMCPGatewayClientStore",
    "OrchidMCPGatewayTokenStore",
    "OrchidOAuthStateStore",
    # ── EXTEND — tools ────────────────────────────────
    "OrchidTool",
    "OrchidToolInput",
    "OrchidToolOutput",
    # ── EXTEND — guardrails ───────────────────────────
    "OrchidGuardrail",
    "OrchidGuardrailAction",
    "OrchidGuardrailChain",
    "OrchidGuardrailContext",
    "OrchidGuardrailDirection",
    "OrchidGuardrailResult",
    # ── EXTEND — content ──────────────────────────────
    "OrchidContentItem",
    "OrchidContentSource",
    # ── EXTEND — registries ───────────────────────────
    "register_guardrail",
    "register_ingestion_strategy",
    "register_post_processor",
    "register_retrieval_strategy",
    "register_query_transformer",
    "register_vector_backend",
    "register_doc_store_backend",
    "register_graph_store_backend",
    "register_sparse_encoder_backend",
    "register_oauth_state_store",
]
