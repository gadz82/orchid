"""
Knowledge-graph primitives for GraphRAG (ADR-026, ADR-028).

Three layered abstractions:

  * :class:`OrchidEntity` / :class:`OrchidEdge` — the data carrying nodes
    and relations across the boundary.
  * :class:`OrchidGraphStore` — the put / lookup / traversal surface that
    a graph_rag retrieval strategy consumes.
  * :class:`OrchidEntityExtractor` — the (LLM- or rule-driven) component
    that converts free text into entities + edges during ingestion.

All four live in ``core/`` so strategies and backends can import them
without risking a cycle.  Concrete impls (in-memory, Neo4j, Memgraph, …)
register themselves in :mod:`orchid_ai.rag.factory` via the
``GRAPH_STORE_BACKEND_REGISTRY`` per ADR-028.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .scopes import OrchidRAGScope


@dataclass(frozen=True)
class OrchidEntity:
    """One node in the knowledge graph.

    ``id`` is the stable identifier the strategy traces; ``type`` and
    ``name`` are display-friendly facets used by entity resolution and
    serialisation.  ``properties`` and ``metadata`` are kept separate so
    backends can index typed properties (numbers, dates) without
    interference from RAG-scope bookkeeping (``tenant_id`` / ``scope`` /
    ``source_doc_id``).
    """

    id: str
    type: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchidEdge:
    """One directed relation between two entities.

    ``relation`` is the relationship label (e.g. ``"supplies"``,
    ``"reports_to"``).  ``properties`` carry edge-typed attributes
    (e.g. ``"weight"``, ``"since"``); ``metadata`` carries the same
    scope bookkeeping as :class:`OrchidEntity`.
    """

    source_id: str
    target_id: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchidGraphStore(ABC):
    """Cypher-free, backend-agnostic graph contract.

    The four methods cover the lifecycle a GraphRAG strategy needs:
    write entities + edges during ingestion, resolve a query into seed
    entities at retrieve time, and walk N hops out from those seeds.
    """

    @abstractmethod
    async def upsert_entities(
        self,
        entities: list[OrchidEntity],
        scope: OrchidRAGScope,
    ) -> None:
        """Insert or update entities in this scope.

        Implementations must be idempotent on ``(scope, entity.id)``.
        """
        ...

    @abstractmethod
    async def upsert_edges(
        self,
        edges: list[OrchidEdge],
        scope: OrchidRAGScope,
    ) -> None:
        """Insert or update edges in this scope.

        Implementations must be idempotent on
        ``(scope, source_id, target_id, relation)``.
        """
        ...

    @abstractmethod
    async def find_entities(
        self,
        *,
        query: str,
        scope: OrchidRAGScope,
        type_filter: list[str] | None = None,
        k: int = 10,
    ) -> list[OrchidEntity]:
        """Resolve free-text mentions in ``query`` to entities.

        Implementations may use exact-name lookup, fuzzy matching, or a
        secondary embedding pass — the ABC does not prescribe a strategy.
        ``type_filter`` narrows the candidate set; ``k`` caps the result.
        """
        ...

    @abstractmethod
    async def neighbours(
        self,
        entity_ids: list[str],
        *,
        scope: OrchidRAGScope,
        max_hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        """Walk up to ``max_hops`` from every seed and return the visited subgraph.

        ``relation_filter`` (when set) restricts traversal to the named
        relationship labels — letting the caller scope the walk to the
        relations meaningful for the question (e.g. ``["reports_to"]`` for
        org-chart queries).
        """
        ...


class OrchidEntityExtractor(ABC):
    """Pull entities + edges out of a chunk of text during ingestion.

    The reference implementation is LLM-driven (structured-output prompt),
    but rule-based and statistical extractors implement the same contract
    so the ingestion pipeline doesn't change when the integrator swaps
    extractors.
    """

    @abstractmethod
    async def extract(
        self,
        text: str,
        *,
        chat_model: Any,
        schema: dict[str, Any] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        """Return ``(entities, edges)`` extracted from ``text``.

        ``schema`` is an optional per-namespace constraint
        (e.g. ``{"entity_types": ["supplier", "product"]}``) that
        implementations may use to narrow the extracted vocabulary.
        ``chat_model`` is duck-typed (``Any``) so this ABC stays free of
        the LangChain ``BaseChatModel`` import in ``core/``.
        """
        ...
