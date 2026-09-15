"""
core/ — Pure abstractions, zero external dependencies.

This package defines the contracts (ABCs, TypedDicts, dataclasses) that every
other package in agents/ depends on.  It must NEVER import anything outside
the Python standard library.

Architectural rule:
    Any import of ``qdrant_client``, ``langchain``, ``langchain_core``,
    ``litellm``, ``httpx``, or any other third-party library inside this
    package is a bug.  The framework's canonical document model is
    :class:`OrchidDocument` — a plain stdlib dataclass.  RAG code that
    needs to interoperate with LangChain goes through the adapter layer
    in :mod:`orchid_ai.rag.adapters`.
"""

from __future__ import annotations

from .agent import OrchidAgent
from .content import OrchidContentItem, OrchidContentSource
from .ingestion_manifest import OrchidIngestionManifest
from .mcp import OrchidMCPClient, OrchidMCPDiscoverable, OrchidMCPToolCaller, OrchidMCPToolResult
from .repository import (
    Document,
    OrchidDocument,
    OrchidSearchResult,
    OrchidVectorReader,
    OrchidVectorStoreRepository,
    OrchidVectorWriter,
)
from .run_config import CONFIG_KEY_AUTH, auth_from_config, with_auth
from .scopes import OrchidRAGScope
from .state import OrchidAgentState, OrchidAuthContext
from .tool import OrchidTool, OrchidToolInput, OrchidToolOutput

__all__ = [
    "CONFIG_KEY_AUTH",
    "Document",
    "OrchidAgent",
    "OrchidAgentState",
    "OrchidAuthContext",
    "OrchidContentItem",
    "OrchidContentSource",
    "OrchidDocument",
    "OrchidIngestionManifest",
    "OrchidMCPClient",
    "OrchidMCPDiscoverable",
    "OrchidMCPToolCaller",
    "OrchidMCPToolResult",
    "OrchidRAGScope",
    "OrchidSearchResult",
    "OrchidTool",
    "OrchidToolInput",
    "OrchidToolOutput",
    "OrchidVectorReader",
    "OrchidVectorStoreRepository",
    "OrchidVectorWriter",
    "auth_from_config",
    "with_auth",
]
