"""
Null implementation of VectorReader — returns empty results.

Used as a placeholder until the Qdrant RAG pipeline is implemented.
Allows the agent graph to run end-to-end without a vector database.
"""

from __future__ import annotations

from ..core.repository import SearchResult, VectorReader
from .scopes import RAGScope


class NullVectorReader(VectorReader):
    """No-op reader that always returns an empty result set."""

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: RAGScope | None = None,
    ) -> list[SearchResult]:
        return []
