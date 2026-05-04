"""
No-op backends — safe placeholders for unconfigured ABCs (ADR-028).

The framework's runtime accessors (``OrchidRuntime.get_reader`` /
``get_doc_store`` / ``get_graph_store``) fall back to these when the
integrator hasn't wired a real backend.  Strategies receive a
fully-typed instance and don't need to handle ``None`` — hierarchical
ingestion against a ``NullDocStore`` degrades to parent-in-metadata,
``graph_rag`` retrieval against a ``NullGraphStore`` falls back to
``SimpleRetrieval`` with a warning, and dense retrieval against
``NullVectorReader`` simply returns an empty result set.

All three impls live next to each other so the ADR-028 import path
``orchid_ai.rag.backends.null.{NullVectorReader, NullDocStore,
NullGraphStore}`` is canonical.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidEdge, OrchidEntity, OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.scopes import OrchidRAGScope


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


class NullDocStore(OrchidDocStore):
    """No-op doc store — accepts writes silently and returns nothing on read.

    The ``is_null`` marker lets hierarchical ingestion (and any other
    strategy that hydrates parents from a docstore) detect this no-op
    fallback and degrade gracefully (e.g. write parent text into chunk
    metadata so retrieval-time hydration still works).
    """

    is_null: ClassVar[bool] = True

    async def put(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        return None

    async def get(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        return None

    async def get_many(self, doc_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        return {}


class NullGraphStore(OrchidGraphStore):
    """No-op graph store — accepts writes silently and returns nothing on read.

    The ``is_null`` marker lets :class:`GraphRAGRetrieval` detect this
    fallback and degrade to :class:`SimpleRetrieval` without crossing
    the ``rag/strategies/`` → ``rag/backends/`` dependency line.
    """

    is_null: ClassVar[bool] = True

    async def upsert_entities(
        self,
        entities: list[OrchidEntity],
        scope: OrchidRAGScope,
    ) -> None:
        return None

    async def upsert_edges(
        self,
        edges: list[OrchidEdge],
        scope: OrchidRAGScope,
    ) -> None:
        return None

    async def find_entities(
        self,
        *,
        query: str,
        scope: OrchidRAGScope,
        type_filter: list[str] | None = None,
        k: int = 10,
    ) -> list[OrchidEntity]:
        return []

    async def neighbours(
        self,
        entity_ids: list[str],
        *,
        scope: OrchidRAGScope,
        max_hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        return ([], [])
