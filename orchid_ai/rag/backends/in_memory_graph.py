"""
``InMemoryGraphStore`` — stdlib-only :class:`OrchidGraphStore`.

Suitable for tests, single-process demos, and integrators who don't
need cross-process persistence.  Keyed by the full
:class:`OrchidRAGScope` tuple so multi-tenant ingestion stays
isolated.

Implementation notes:

* ``find_entities`` does substring + exact-name lookups — no fuzzy
  matching.  Custom backends (Neo4j, Memgraph, AGE) substitute their
  native full-text or vector search.
* ``neighbours`` is an undirected BFS up to ``max_hops`` from every
  seed.  Edges are added to the result regardless of direction; the
  edge's ``source_id`` / ``target_id`` are preserved on the returned
  :class:`OrchidEdge` so callers can reconstruct the original direction
  when serialising.
* The store is **not** thread-safe across event loops.  Single-process
  workloads are the target audience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.graph_store import OrchidEdge, OrchidEntity, OrchidGraphStore
from ...core.scopes import OrchidRAGScope


@dataclass
class _ScopeState:
    entities: dict[str, OrchidEntity] = field(default_factory=dict)
    out_edges: dict[str, list[OrchidEdge]] = field(default_factory=dict)
    in_edges: dict[str, list[OrchidEdge]] = field(default_factory=dict)


def _scope_key(scope: OrchidRAGScope) -> tuple[str, str, str, str]:
    return (scope.tenant_id, scope.user_id, scope.chat_id, scope.agent_id)


def _edge_key(edge: OrchidEdge) -> tuple[str, str, str]:
    return (edge.source_id, edge.target_id, edge.relation)


class InMemoryGraphStore(OrchidGraphStore):
    """Dict-backed knowledge-graph store.

    Per-scope isolation: every operation is keyed by the
    ``(tenant_id, user_id, chat_id, agent_id)`` tuple.  Reading from a
    scope that's never been written to returns empty results
    (no implicit cross-scope leakage).
    """

    def __init__(self) -> None:
        self._scopes: dict[tuple[str, str, str, str], _ScopeState] = {}

    # ── Mutations ─────────────────────────────────────────────

    async def upsert_entities(
        self,
        entities: list[OrchidEntity],
        scope: OrchidRAGScope,
    ) -> None:
        state = self._scopes.setdefault(_scope_key(scope), _ScopeState())
        for entity in entities:
            state.entities[entity.id] = entity

    async def upsert_edges(
        self,
        edges: list[OrchidEdge],
        scope: OrchidRAGScope,
    ) -> None:
        state = self._scopes.setdefault(_scope_key(scope), _ScopeState())
        for edge in edges:
            seen_keys = {_edge_key(e) for e in state.out_edges.get(edge.source_id, [])}
            if _edge_key(edge) in seen_keys:
                continue  # Idempotent — same triple replaces nothing
            state.out_edges.setdefault(edge.source_id, []).append(edge)
            state.in_edges.setdefault(edge.target_id, []).append(edge)

    # ── Queries ──────────────────────────────────────────────

    async def find_entities(
        self,
        *,
        query: str,
        scope: OrchidRAGScope,
        type_filter: list[str] | None = None,
        k: int = 10,
    ) -> list[OrchidEntity]:
        state = self._scopes.get(_scope_key(scope))
        if state is None:
            return []

        type_set = set(type_filter) if type_filter else None
        q_lower = query.lower().strip()
        if not q_lower:
            return []

        scored: list[tuple[float, OrchidEntity]] = []
        for entity in state.entities.values():
            if type_set is not None and entity.type not in type_set:
                continue
            score = _match_score(q_lower, entity)
            if score > 0:
                scored.append((score, entity))

        # Higher score first; stable on ties via id ordering for
        # deterministic test output.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [entity for _, entity in scored[:k]]

    async def neighbours(
        self,
        entity_ids: list[str],
        *,
        scope: OrchidRAGScope,
        max_hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        state = self._scopes.get(_scope_key(scope))
        if state is None or max_hops < 0:
            return ([], [])

        relation_set = set(relation_filter) if relation_filter else None
        visited_entities: dict[str, OrchidEntity] = {}
        visited_edge_keys: set[tuple[str, str, str]] = set()
        visited_edges: list[OrchidEdge] = []

        # Seed entities go into the visited set first so they appear
        # in the result even if they have no outgoing or incoming edges.
        current_layer: set[str] = set()
        for entity_id in entity_ids:
            if entity_id in state.entities:
                visited_entities[entity_id] = state.entities[entity_id]
                current_layer.add(entity_id)

        # BFS — undirected (combine outgoing + incoming) up to ``max_hops``.
        for _ in range(max_hops):
            next_layer: set[str] = set()
            for entity_id in current_layer:
                outgoing = state.out_edges.get(entity_id, [])
                incoming = state.in_edges.get(entity_id, [])
                for edge in (*outgoing, *incoming):
                    if relation_set is not None and edge.relation not in relation_set:
                        continue
                    key = _edge_key(edge)
                    if key in visited_edge_keys:
                        continue
                    visited_edge_keys.add(key)
                    visited_edges.append(edge)

                    other_id = edge.target_id if edge.source_id == entity_id else edge.source_id
                    if other_id not in visited_entities and other_id in state.entities:
                        visited_entities[other_id] = state.entities[other_id]
                        next_layer.add(other_id)
            if not next_layer:
                break
            current_layer = next_layer

        return (list(visited_entities.values()), visited_edges)


def _match_score(q_lower: str, entity: OrchidEntity) -> float:
    """Exact-name → 1.0, exact-id → 0.95, name substring → 0.6,
    id substring → 0.4, otherwise 0.0."""
    name_lower = (entity.name or "").lower()
    id_lower = entity.id.lower()
    if name_lower and name_lower == q_lower:
        return 1.0
    if id_lower == q_lower:
        return 0.95
    if name_lower and q_lower in name_lower:
        return 0.6
    if q_lower in id_lower:
        return 0.4
    return 0.0
