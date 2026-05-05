"""
Vector store abstractions — Interface Segregation.

OrchidVectorReader: agents that only retrieve.
OrchidVectorWriter: indexers that only write.
OrchidVectorStoreRepository: combines both (for components that need full access).

Uses LangChain's ``Document`` as the standard document model:

    from langchain_core.documents import Document
    doc = Document(page_content="text", metadata={"scope": "tenant"}, id="doc-1")

Architectural rule:
    No agent, tool, or pipeline may import QdrantClient, opensearchpy,
    or any other concrete vector DB client.  All access goes through
    these ABCs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from langchain_core.documents import Document

from .scopes import OrchidRAGScope  # noqa: F401 — used in type annotations
from .sparse import OrchidSparseVector

logger = logging.getLogger(__name__)


@dataclass
class OrchidSearchResult:
    """A document with its relevance score."""

    document: Document
    score: float  # 0.0 → 1.0


class OrchidVectorReader(ABC):
    """Read-only access to the vector store (for agents)."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        """Return the k most relevant documents for the query.

        ``metadata_filters`` follows the operator mini-language
        (``"key": value`` for exact match, ``"key": [v1, v2]``
        for match-any, ``"key": {"gte": ..., "lte": ...}`` for range,
        ``"key": {"contains": v}`` for array contains, ``"key": {"not": v}``
        for negation, ``"_<backend>": {...}`` for backend-namespaced
        extras).  Backends that don't support filtering ignore the
        kwarg silently — the agent's retrieval flow remains correct,
        just unfiltered.
        """
        ...

    async def retrieve_sparse(
        self,
        query_sparse: OrchidSparseVector,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        """Retrieve via the sparse / lexical lane.

        Default body raises :class:`NotImplementedError` so existing
        custom readers stay LSP-compliant — :class:`HybridRetrieval`
        catches the error and degrades to dense-only with a warning.

        Backends that support hybrid (Qdrant; OpenSearch / Weaviate /
        pgvector via integrator-supplied impls) override this method
        to translate the sparse vector into the backend's native
        sparse-search primitive.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sparse retrieval. "
            "Hybrid search requires a backend with a sparse lane "
            "(e.g. Qdrant with named sparse vectors)."
        )

    async def lookup_cached_tool_results(
        self,
        namespace: str,
        scope: OrchidRAGScope,
        tool_name: str,
        min_injected_at: float,
    ) -> str | None:
        """Lookup cached tool results by metadata. Returns content or None.

        Default implementation returns None (no cache support).
        Backends with payload filtering (e.g. Qdrant) override this.
        """
        return None


class OrchidVectorWriter(ABC):
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


class OrchidVectorStoreAdmin(ABC):
    """Administrative operations on the vector store (collection management).

    Implementations should handle idempotent collection creation.
    This interface lets the API layer manage collections without
    depending on a concrete backend (e.g. QdrantRepository).
    """

    @abstractmethod
    async def ensure_collections(self, namespaces: list[str]) -> None:
        """Ensure that collections/indices exist for the given namespaces."""
        ...


class OrchidVectorStoreRepository(OrchidVectorReader, OrchidVectorWriter, OrchidVectorStoreAdmin, ABC):
    """
    Full read+write+admin access — used by components that need all
    capabilities (e.g. the Qdrant backend).
    """

    #: Set to ``True`` in subclasses that implement :meth:`promote_scope`.
    #: Callers use this to distinguish "promoted 0 points" from
    #: "backend doesn't support promotion" without introspecting the class.
    supports_scope_promotion: ClassVar[bool] = False

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

        The base implementation is a no-op returning ``0`` so callers can
        safely invoke it on any :class:`OrchidVectorStoreRepository` (LSP
        compliance).  Check :attr:`supports_scope_promotion` beforehand
        when the caller must distinguish "no points matched" from
        "backend does not support promotion".
        """
        logger.debug("[%s] promote_scope() not implemented — returning 0", type(self).__name__)
        return 0
