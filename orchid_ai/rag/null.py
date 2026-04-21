"""
Null implementation of OrchidVectorReader — returns empty results.

Used as a placeholder until the Qdrant RAG pipeline is implemented.
Allows the agent graph to run end-to-end without a vector database.
"""

from __future__ import annotations

from ..core.repository import OrchidSearchResult, OrchidVectorReader
from .scopes import OrchidRAGScope


class NullVectorReader(OrchidVectorReader):
    """No-op reader that always returns an empty result set."""

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
    ) -> list[OrchidSearchResult]:
        return []
