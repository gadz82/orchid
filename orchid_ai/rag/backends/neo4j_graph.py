"""
``Neo4jGraphStore`` — :class:`OrchidGraphStore` backed by Neo4j (ADR-026).

Behind the optional ``neo4j`` extra: ``pip install orchid-ai[neo4j]``.
The constructor eagerly checks for the ``neo4j`` driver — surfacing
the missing-extra error at registry lookup time rather than burying
it in the first ``upsert_entities`` call.

The Cypher mapping:

* Entities → ``(:Entity {id, type, name, scope, properties})``.
* Edges   → ``(:Entity)-[:RELATES_TO {relation, scope, properties}]->(:Entity)``.
* Scope is encoded as a JSON-serialised property so multi-tenant
  isolation works without dynamic label creation.

The implementation is intentionally minimal — integrators with
existing Neo4j deployments often subclass to wire their own labels,
indexes, and Cypher dialects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...core.graph_store import OrchidEdge, OrchidEntity, OrchidGraphStore
from ...core.scopes import OrchidRAGScope

logger = logging.getLogger(__name__)


_INSTALL_HINT = "Neo4jGraphStore requires the 'neo4j' extra. Install with: pip install orchid-ai[neo4j]"


def _scope_dict(scope: OrchidRAGScope) -> dict[str, str]:
    return {
        "tenant_id": scope.tenant_id,
        "user_id": scope.user_id,
        "chat_id": scope.chat_id,
        "agent_id": scope.agent_id,
    }


def _scope_key(scope: OrchidRAGScope) -> str:
    """Stable JSON encoding so ``WHERE e.scope = $scope`` matches."""
    return json.dumps(_scope_dict(scope), sort_keys=True)


class Neo4jGraphStore(OrchidGraphStore):
    """Neo4j-backed knowledge-graph store.

    Parameters
    ----------
    url : str
        Bolt URL (e.g. ``bolt://localhost:7687``).
    user, password : str
        Neo4j credentials.  When both empty, the driver runs without
        authentication (test setups).
    database : str
        Target database name.  Default ``"neo4j"``.
    driver : Any
        Optional pre-built async driver — used by tests to inject a
        mock.  Construction skips the network handshake when supplied.

    Raises
    ------
    ImportError
        When the ``neo4j`` extra is not installed.
    """

    def __init__(
        self,
        *,
        url: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        database: str = "neo4j",
        driver: Any | None = None,
    ) -> None:
        try:
            import neo4j  # noqa: F401
        except ImportError as exc:  # pragma: no cover — exercised via test_rag_neo4j_graph
            raise ImportError(_INSTALL_HINT) from exc

        self._url = url
        self._user = user
        self._password = password
        self._database = database
        self._driver: Any | None = driver

    def _ensure_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        from neo4j import AsyncGraphDatabase

        auth = (self._user, self._password) if (self._user or self._password) else None
        self._driver = AsyncGraphDatabase.driver(self._url, auth=auth)
        return self._driver

    async def close(self) -> None:
        """Close the underlying driver if owned by this store."""
        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("[Neo4jGraphStore] driver close failed: %s", exc)
            self._driver = None

    # ── Mutations ─────────────────────────────────────────────

    async def upsert_entities(
        self,
        entities: list[OrchidEntity],
        scope: OrchidRAGScope,
    ) -> None:
        if not entities:
            return
        driver = self._ensure_driver()
        scope_key = _scope_key(scope)
        async with driver.session(database=self._database) as session:
            for entity in entities:
                await session.run(
                    "MERGE (e:Entity {id: $id, scope: $scope}) "
                    "SET e.type = $type, e.name = $name, "
                    "    e.properties = $properties, e.metadata = $metadata",
                    id=entity.id,
                    scope=scope_key,
                    type=entity.type,
                    name=entity.name,
                    properties=json.dumps(entity.properties, default=str),
                    metadata=json.dumps(entity.metadata, default=str),
                )

    async def upsert_edges(
        self,
        edges: list[OrchidEdge],
        scope: OrchidRAGScope,
    ) -> None:
        if not edges:
            return
        driver = self._ensure_driver()
        scope_key = _scope_key(scope)
        async with driver.session(database=self._database) as session:
            for edge in edges:
                await session.run(
                    "MATCH (s:Entity {id: $source_id, scope: $scope}) "
                    "MATCH (t:Entity {id: $target_id, scope: $scope}) "
                    "MERGE (s)-[r:RELATES_TO {relation: $relation, scope: $scope}]->(t) "
                    "SET r.properties = $properties, r.metadata = $metadata",
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    scope=scope_key,
                    relation=edge.relation,
                    properties=json.dumps(edge.properties, default=str),
                    metadata=json.dumps(edge.metadata, default=str),
                )

    # ── Queries ──────────────────────────────────────────────

    async def find_entities(
        self,
        *,
        query: str,
        scope: OrchidRAGScope,
        type_filter: list[str] | None = None,
        k: int = 10,
    ) -> list[OrchidEntity]:
        driver = self._ensure_driver()
        scope_key = _scope_key(scope)
        type_clause = ""
        if type_filter:
            type_clause = " AND e.type IN $types"
        cypher = (
            "MATCH (e:Entity {scope: $scope}) "
            "WHERE (toLower(e.name) CONTAINS toLower($query) "
            "       OR toLower(e.id) CONTAINS toLower($query))"
            f"{type_clause} "
            "RETURN e LIMIT $k"
        )
        async with driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                query=query,
                scope=scope_key,
                types=list(type_filter or []),
                k=k,
            )
            return [self._record_to_entity(rec["e"]) async for rec in result]

    async def neighbours(
        self,
        entity_ids: list[str],
        *,
        scope: OrchidRAGScope,
        max_hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        if not entity_ids:
            return ([], [])
        driver = self._ensure_driver()
        scope_key = _scope_key(scope)
        relation_clause = ""
        if relation_filter:
            relation_clause = " WHERE all(rel IN relationships(p) WHERE rel.relation IN $relations)"
        cypher = (
            "MATCH p = (s:Entity {scope: $scope})-[*0.."
            f"{int(max_hops)}"
            "]-(t:Entity {scope: $scope}) "
            "WHERE s.id IN $seeds"
            f"{relation_clause} "
            "RETURN nodes(p) AS nodes, relationships(p) AS rels"
        )
        seen_entities: dict[str, OrchidEntity] = {}
        seen_edges: dict[tuple[str, str, str], OrchidEdge] = {}
        async with driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                seeds=list(entity_ids),
                scope=scope_key,
                relations=list(relation_filter or []),
            )
            async for record in result:
                for node in record["nodes"]:
                    entity = self._record_to_entity(node)
                    seen_entities.setdefault(entity.id, entity)
                for rel in record["rels"]:
                    edge = self._record_to_edge(rel)
                    seen_edges.setdefault((edge.source_id, edge.target_id, edge.relation), edge)
        return (list(seen_entities.values()), list(seen_edges.values()))

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _record_to_entity(record: Any) -> OrchidEntity:
        return OrchidEntity(
            id=record["id"],
            type=record.get("type", "") or "",
            name=record.get("name", "") or "",
            properties=json.loads(record.get("properties", "{}") or "{}"),
            metadata=json.loads(record.get("metadata", "{}") or "{}"),
        )

    @staticmethod
    def _record_to_edge(record: Any) -> OrchidEdge:
        # ``record`` is a Neo4j relationship — has ``.start_node`` /
        # ``.end_node`` / ``.relation`` payload.
        start = record.start_node
        end = record.end_node
        return OrchidEdge(
            source_id=start["id"],
            target_id=end["id"],
            relation=record.get("relation", "") or "",
            properties=json.loads(record.get("properties", "{}") or "{}"),
            metadata=json.loads(record.get("metadata", "{}") or "{}"),
        )
