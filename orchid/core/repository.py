"""
Vector store abstractions — Interface Segregation (ADR-008).

VectorReader: agents that only retrieve.
VectorWriter: indexers that only write.
VectorStoreRepository: combines both (for components that need full access).

Architectural rule:
    No agent, tool, or pipeline may import QdrantClient, opensearchpy,
    or any other concrete vector DB client.  All access goes through
    these ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..rag.scopes import RAGScope


@dataclass
class Document:
    """A piece of content indexed in the vector store."""

    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A document with its relevance score."""

    document: Document
    score: float  # 0.0 → 1.0


class VectorReader(ABC):
    """Read-only access to the vector store (for agents)."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: RAGScope | None = None,
    ) -> list[SearchResult]:
        """Return the k most relevant documents for the query."""
        ...

    async def lookup_cached_tool_results(
        self,
        namespace: str,
        scope: RAGScope,
        tool_name: str,
        min_injected_at: float,
    ) -> str | None:
        """Lookup cached tool results by metadata. Returns content or None.

        Default implementation returns None (no cache support).
        Backends with payload filtering (e.g. Qdrant) override this.
        """
        return None


class VectorWriter(ABC):
    """Write-only access to the vector store (for indexers)."""

    @abstractmethod
    async def index(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Index documents into the namespace."""
        ...

    @abstractmethod
    async def upsert(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Insert or update documents (idempotent)."""
        ...

    @abstractmethod
    async def delete(
        self,
        document_ids: list[str],
        namespace: str,
    ) -> None:
        """Remove documents from the namespace."""
        ...


class VectorStoreAdmin(ABC):
    """Administrative operations on the vector store (collection management).

    Implementations should handle idempotent collection creation.
    This interface lets the API layer manage collections without
    depending on a concrete backend (e.g. QdrantRepository).
    """

    @abstractmethod
    async def ensure_collections(self, namespaces: list[str]) -> None:
        """Ensure that collections/indices exist for the given namespaces."""
        ...


class VectorStoreRepository(VectorReader, VectorWriter, VectorStoreAdmin, ABC):
    """
    Full read+write+admin access — used by components that need all
    capabilities (e.g. the Qdrant backend).
    """

    async def promote_scope(
        self,
        *,
        namespace: str,
        source_filter: Any,
        new_scope_fields: dict,
    ) -> int:
        """
        Move data to a broader scope (e.g. chat → user for sharing).

        Parameters
        ----------
        namespace : str
            Vector store collection name.
        source_filter : Any
            Backend-specific filter identifying the source points.
        new_scope_fields : dict
            Metadata fields to apply to the duplicated points.

        Returns
        -------
        int
            Number of points promoted.

        Subclasses must override.  Default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support promote_scope(). Override this method to enable chat sharing."
        )
