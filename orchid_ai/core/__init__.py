"""
core/ — Pure abstractions, zero external dependencies.

This package defines the contracts (ABCs, TypedDicts, dataclasses) that every
other package in agents/ depends on.  It must NEVER import anything outside
the Python standard library.

Architectural rule (ADR-008):
    Any import of qdrant_client, langchain, litellm, httpx, or any other
    third-party library inside this package is a bug.
"""

from .state import AgentState, AuthContext
from .agent import BaseAgent
from .repository import (
    Document,
    SearchResult,
    VectorReader,
    VectorWriter,
    VectorStoreRepository,
)
from .scopes import RAGScope
from .mcp import MCPClient, MCPDiscoverable, MCPToolCaller, MCPToolResult

__all__ = [
    "AgentState",
    "AuthContext",
    "BaseAgent",
    "Document",
    "RAGScope",
    "SearchResult",
    "VectorReader",
    "VectorWriter",
    "VectorStoreRepository",
    "MCPClient",
    "MCPDiscoverable",
    "MCPToolCaller",
    "MCPToolResult",
]
