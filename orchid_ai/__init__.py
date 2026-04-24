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

__version__ = "0.0.0"

from .checkpointing.factory import build_checkpointer, shutdown_checkpointer
from .config.loader import load_config
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
from .core.identity import OrchidIdentityError, OrchidIdentityResolver
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
from .core.repository import (
    Document,
    OrchidSearchResult,
    OrchidVectorReader,
    OrchidVectorStoreAdmin,
    OrchidVectorStoreRepository,
    OrchidVectorWriter,
)
from .core.state import OrchidAgentState, OrchidAuthContext
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
from .mcp.oauth_state import (
    OrchidInMemoryOAuthStateStore,
    OrchidOAuthPendingState,
    OrchidOAuthStateStore,
    build_oauth_state_store,
    register_oauth_state_store,
)
from .observability.callbacks import OrchidMetricsHandler
from .orchid import Orchid, OrchidInvokeResult, OrchidPendingApproval
from .persistence.base import OrchidChatStorage
from .persistence.factory import build_chat_storage
from .persistence.mcp_client_registration_factory import build_mcp_client_registration_store
from .persistence.mcp_client_registration_sqlite import OrchidSQLiteMCPClientRegistrationStore
from .persistence.mcp_token_factory import build_mcp_token_store
from .persistence.mcp_token_sqlite import OrchidSQLiteMCPTokenStore
from .persistence.sqlite import OrchidSQLiteChatStorage
from .plugins import iter_entry_point_plugins
from .rag.factory import build_reader
from .rag.scopes import OrchidRAGScope
from .runtime import OrchidRuntime

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
    "OrchidIdentityError",
    "OrchidIdentityResolver",
    "OrchidMCPClient",
    "OrchidMCPClientRegistrationStore",
    "OrchidMCPDiscoverable",
    "OrchidMCPToolCaller",
    "OrchidMCPTokenStore",
    "OrchidVectorReader",
    "OrchidVectorStoreAdmin",
    "OrchidVectorStoreRepository",
    "OrchidVectorWriter",
    # ── Data / result types ───────────────────────────
    "Document",
    "OrchidMCPAuthDiscovery",
    "OrchidMCPAuthRegistry",
    "OrchidMCPAuthRequiredError",
    "OrchidMCPClientRegistration",
    "OrchidMCPDiscoveryError",
    "OrchidMCPOAuthServerInfo",
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
    "OrchidSQLiteChatStorage",
    "OrchidSQLiteMCPClientRegistrationStore",
    "OrchidSQLiteMCPTokenStore",
    # ── Runtime + observability ───────────────────────
    "OrchidRuntime",
    "OrchidMetricsHandler",
    # ── Factories ─────────────────────────────────────
    "build_chat_model",
    "build_chat_storage",
    "build_checkpointer",
    "build_graph",
    "build_mcp_client_registration_store",
    "build_mcp_token_store",
    "build_reader",
    "iter_entry_point_plugins",
    "load_config",
    "shutdown_checkpointer",
]
