"""
``GraphRAGRetrieval`` — knowledge-graph-backed retrieval.

The pipeline is four steps:

  1. **Entity resolution** — resolve the user's query into seed entities
     via :meth:`OrchidGraphStore.find_entities`.
  2. **Sub-graph walk** — :meth:`OrchidGraphStore.neighbours` walks
     up to ``max_hops`` from every seed, returning the visited
     entities + edges.
  3. **Vector retrieval** — fetch text chunks via the standard
     :meth:`OrchidVectorReader.retrieve` so the LLM also has the raw
     document evidence.
  4. **Sub-graph serialisation** — encode the visited triples as a
     plain-text block and prepend it as a synthetic
     :class:`OrchidSearchResult` (``source: graph_rag``).  The
     ``fuse_with_vectors`` flag controls whether vector hits follow.

When ``graph_store`` is missing or the no-op
:class:`NullGraphStore` is wired, the strategy logs a one-line
warning and falls back to :class:`SimpleRetrieval` so an integrator
who specifies ``strategy: graph_rag`` without wiring a store still
gets sensible results.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from langchain_core.documents import Document

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidEdge, OrchidEntity, OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from ...core.scopes import OrchidRAGScope
from .simple import SimpleRetrieval

logger = logging.getLogger(__name__)


_EntitySerializer = Callable[[list[OrchidEntity], list[OrchidEdge]], str]


class GraphRAGRetrieval(OrchidRetrievalStrategy):
    """Knowledge-graph-augmented retrieval."""

    def __init__(
        self,
        *,
        max_hops: int = 2,
        fuse_with_vectors: bool = True,
        relation_filter: list[str] | None = None,
        seed_k: int = 10,
        entity_serializer: _EntitySerializer | None = None,
    ) -> None:
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0; got {max_hops}")
        if seed_k < 1:
            raise ValueError(f"seed_k must be >= 1; got {seed_k}")
        self._max_hops = max_hops
        self._fuse_with_vectors = fuse_with_vectors
        self._relation_filter = list(relation_filter) if relation_filter else None
        self._seed_k = seed_k
        self._serialiser = entity_serializer or _default_serialise

    @classmethod
    def from_config(cls, config: Any) -> "GraphRAGRetrieval":
        graph_cfg = getattr(config, "graph", None) if config is not None else None
        if graph_cfg is None:
            return cls()
        return cls(
            max_hops=getattr(graph_cfg, "max_hops", 2),
            fuse_with_vectors=getattr(graph_cfg, "fuse_with_vectors", True),
            relation_filter=list(getattr(graph_cfg, "relation_filter", []) or []),
        )

    async def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        scope: OrchidRAGScope,
        k: int,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        doc_store: OrchidDocStore | None = None,
        transformers: list[OrchidQueryTransformer] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        if graph_store is None or getattr(graph_store, "is_null", False):
            logger.warning("[GraphRAGRetrieval] no graph_store wired — falling back to SimpleRetrieval")
            return await SimpleRetrieval().retrieve(
                query=query,
                namespace=namespace,
                scope=scope,
                k=k,
                reader=reader,
                metadata_filters=metadata_filters,
            )

        # Step 1 — resolve seed entities mentioned in the query.
        seed_entities = await graph_store.find_entities(query=query, scope=scope, k=self._seed_k)
        if not seed_entities:
            logger.debug("[GraphRAGRetrieval] no seed entities resolved for %r — vector-only", query)
            return await SimpleRetrieval().retrieve(
                query=query,
                namespace=namespace,
                scope=scope,
                k=k,
                reader=reader,
                metadata_filters=metadata_filters,
            )

        # Step 2 — walk the graph.
        entities, edges = await graph_store.neighbours(
            [e.id for e in seed_entities],
            scope=scope,
            max_hops=self._max_hops,
            relation_filter=self._relation_filter,
        )

        # Step 3 — fetch text chunks via the vector lane.
        chunk_results: list[OrchidSearchResult] = []
        if self._fuse_with_vectors:
            chunk_results = await reader.retrieve(
                query=query,
                namespace=namespace,
                k=max(k * 2, k + 1),
                scope=scope,
                metadata_filters=metadata_filters,
            )

        # Step 4 — serialise the sub-graph as a synthetic result.
        graph_text = self._serialiser(entities, edges)
        graph_doc_id = f"graph::{hashlib.sha256(graph_text.encode()).hexdigest()[:16]}"
        graph_result = OrchidSearchResult(
            document=Document(
                id=graph_doc_id,
                page_content=graph_text,
                metadata={
                    "source": "graph_rag",
                    "scope": "chat_shared",
                    "tenant_id": scope.tenant_id,
                    "user_id": scope.user_id,
                    "chat_id": scope.chat_id,
                    "agent_id": scope.agent_id,
                    "entity_count": len(entities),
                    "edge_count": len(edges),
                    "mentioned_entities": sorted({e.id for e in entities}),
                },
            ),
            score=1.0,
        )

        if not self._fuse_with_vectors:
            return [graph_result]

        # Prepend the graph context, then top up with vector hits up to k.
        merged: list[OrchidSearchResult] = [graph_result]
        for hit in chunk_results:
            if hit.document.id == graph_doc_id:
                continue
            merged.append(hit)
            if len(merged) >= k:
                break
        return merged[:k]


def _default_serialise(
    entities: list[OrchidEntity],
    edges: list[OrchidEdge],
) -> str:
    """Plain-text serialisation: ``Entity (type) [-relation-> Other]``.

    Pinned by tests so integrators see exactly what reaches the LLM.
    Override via the ``entity_serializer`` constructor kwarg for
    domain-specific formats (RDF turtle, JSON-LD, ...).
    """
    if not entities and not edges:
        return "[Knowledge graph context — empty]"

    by_id: dict[str, OrchidEntity] = {e.id: e for e in entities}

    lines: list[str] = ["[Knowledge graph context]"]
    if entities:
        lines.append("Entities:")
        for entity in sorted(entities, key=lambda e: e.id):
            display = entity.name or entity.id
            lines.append(f"  - {display} ({entity.type or 'unknown'}) — id={entity.id}")

    if edges:
        lines.append("Relations:")
        for edge in sorted(edges, key=lambda e: (e.source_id, e.relation, e.target_id)):
            src = by_id.get(edge.source_id)
            tgt = by_id.get(edge.target_id)
            src_label = src.name if src and src.name else edge.source_id
            tgt_label = tgt.name if tgt and tgt.name else edge.target_id
            lines.append(f"  - {src_label} -[{edge.relation}]-> {tgt_label}")

    return "\n".join(lines)
