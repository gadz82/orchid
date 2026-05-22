"""
core/ — Pure abstractions, zero external dependencies.

This package defines the contracts (ABCs, TypedDicts, dataclasses) that every
other package in agents/ depends on.  It must NEVER import anything outside
the Python standard library.

Architectural rule:
    Any import of qdrant_client, langchain, litellm, httpx, or any other
    third-party library inside this package is a bug.
"""

from __future__ import annotations

from .state import OrchidAgentState, OrchidAuthContext
from .agent import OrchidAgent
from .repository import (
    Document,
    OrchidSearchResult,
    OrchidVectorReader,
    OrchidVectorWriter,
    OrchidVectorStoreRepository,
)
from .scopes import OrchidRAGScope
from .mcp import OrchidMCPClient, OrchidMCPDiscoverable, OrchidMCPToolCaller, OrchidMCPToolResult
from .content import OrchidContentItem, OrchidContentSource

__all__ = [
    "OrchidAgentState",
    "OrchidAuthContext",
    "OrchidAgent",
    "Document",
    "OrchidRAGScope",
    "OrchidSearchResult",
    "OrchidVectorReader",
    "OrchidVectorWriter",
    "OrchidVectorStoreRepository",
    "OrchidMCPClient",
    "OrchidMCPDiscoverable",
    "OrchidMCPToolCaller",
    "OrchidMCPToolResult",
    "OrchidContentItem",
    "OrchidContentSource",
]
