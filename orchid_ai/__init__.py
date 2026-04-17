"""
Orchid — platform-agnostic multi-agent AI framework.

Public SDK surface — import the most common types directly:

    from orchid_ai import BaseAgent, AuthContext, build_graph, load_config
"""

from __future__ import annotations

__version__ = "0.0.0"

from .bootstrap import BootstrapResult, build_runtime, teardown_runtime
from .checkpointing.factory import build_checkpointer, shutdown_checkpointer
from .client import InvokeResult, OrchidClient, PendingApproval
from .config.loader import load_config
from .observability.callbacks import OrchidMetricsHandler
from .core.agent import BaseAgent
from .core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailChain,
    GuardrailContext,
    GuardrailDirection,
    GuardrailResult,
)
from .core.identity import IdentityError, IdentityResolver
from .llm_factory import build_chat_model
from .core.mcp import (
    MCPAuthRequiredError,
    MCPClient,
    MCPDiscoverable,
    MCPTokenRecord,
    MCPTokenStore,
    MCPToolCaller,
    MCPToolResult,
)
from .core.repository import (
    Document,
    SearchResult,
    VectorReader,
    VectorStoreAdmin,
    VectorStoreRepository,
    VectorWriter,
)
from .core.state import AgentState, AuthContext
from .graph.graph import build_graph
from .graph.supervisor import RoutingDecision
from .guardrails import build_guardrail_chain, register_guardrail
from .mcp.auth_registry import MCPAuthRegistry, MCPOAuthServerInfo
from .mcp.oauth_state import (
    InMemoryOAuthStateStore,
    OAuthPendingState,
    OAuthStateStore,
    build_oauth_state_store,
    register_oauth_state_store,
)
from .persistence.base import ChatStorage
from .persistence.factory import build_chat_storage
from .persistence.mcp_token_factory import build_mcp_token_store
from .persistence.mcp_token_sqlite import SQLiteMCPTokenStore
from .persistence.sqlite import SQLiteChatStorage
from .plugins import iter_entry_point_plugins
from .rag.factory import build_reader
from .rag.scopes import RAGScope
from .runtime import OrchidRuntime

# ``utils.import_class`` is intentionally NOT re-exported on the top-level
# ``orchid_ai`` namespace — it's an implementation detail used by the
# framework's own factories (chat storage, MCP store, checkpointers,
# identity resolver) and can change without notice.  Integrators who
# genuinely need dynamic dotted-path resolution should import it
# explicitly from :mod:`orchid_ai.utils`.

__all__ = [
    # core ABCs
    "AgentState",
    "AuthContext",
    "BaseAgent",
    "ChatStorage",
    "MCPAuthRegistry",
    "MCPAuthRequiredError",
    "MCPOAuthServerInfo",
    "MCPTokenRecord",
    "MCPTokenStore",
    # MCP OAuth state store
    "InMemoryOAuthStateStore",
    "OAuthPendingState",
    "OAuthStateStore",
    "build_oauth_state_store",
    "register_oauth_state_store",
    "SQLiteChatStorage",
    "SQLiteMCPTokenStore",
    "Document",
    "Guardrail",
    "GuardrailAction",
    "GuardrailChain",
    "GuardrailContext",
    "GuardrailDirection",
    "GuardrailResult",
    "IdentityError",
    "IdentityResolver",
    "InvokeResult",
    "OrchidClient",
    "PendingApproval",
    # bootstrap
    "BootstrapResult",
    "build_runtime",
    "teardown_runtime",
    "build_chat_model",
    "MCPClient",
    "MCPDiscoverable",
    "MCPToolCaller",
    "MCPToolResult",
    "RAGScope",
    "SearchResult",
    "VectorReader",
    "VectorStoreAdmin",
    "VectorStoreRepository",
    "VectorWriter",
    # runtime
    "OrchidRuntime",
    # observability
    "OrchidMetricsHandler",
    # checkpointing
    "build_checkpointer",
    "shutdown_checkpointer",
    # factories
    "build_chat_storage",
    "build_mcp_token_store",
    "build_graph",
    "RoutingDecision",
    "build_guardrail_chain",
    "build_reader",
    "iter_entry_point_plugins",
    "load_config",
    "register_guardrail",
]
