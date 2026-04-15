"""
Orchid — platform-agnostic multi-agent AI framework.

Public SDK surface — import the most common types directly:

    from orchid_ai import BaseAgent, AuthContext, build_graph, load_config
"""

from __future__ import annotations

__version__ = "0.0.0"

from .config.loader import load_config
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
from .core.llm_provider import LLMProvider
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
from .guardrails import build_guardrail_chain, register_guardrail
from .mcp.auth_registry import MCPAuthRegistry, MCPOAuthServerInfo
from .persistence.base import ChatStorage
from .persistence.factory import build_chat_storage
from .persistence.mcp_token_factory import build_mcp_token_store
from .persistence.mcp_token_sqlite import SQLiteMCPTokenStore
from .persistence.sqlite import SQLiteChatStorage
from .rag.factory import build_reader
from .rag.scopes import RAGScope
from .runtime import OrchidRuntime
from .utils import import_class

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
    "LLMProvider",
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
    # factories
    "build_chat_storage",
    "build_mcp_token_store",
    "build_graph",
    "build_guardrail_chain",
    "build_reader",
    "import_class",
    "load_config",
    "register_guardrail",
]
