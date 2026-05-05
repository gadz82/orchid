"""Tests for ``InMemoryGraphStore``."""

from __future__ import annotations

import pytest

from orchid_ai.core.graph_store import OrchidEdge, OrchidEntity
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore


def _scope(suffix: str = "") -> OrchidRAGScope:
    return OrchidRAGScope(
        tenant_id=f"t1{suffix}",
        user_id=f"u1{suffix}",
        chat_id=f"c1{suffix}",
        agent_id=f"a1{suffix}",
    )


def _entity(eid: str, type_: str = "person", name: str | None = None) -> OrchidEntity:
    return OrchidEntity(id=eid, type=type_, name=name or eid, properties={}, metadata={})


def _edge(src: str, tgt: str, relation: str = "knows") -> OrchidEdge:
    return OrchidEdge(source_id=src, target_id=tgt, relation=relation, properties={}, metadata={})


class TestUpsert:
    @pytest.mark.asyncio
    async def test_round_trip_entities(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a"), _entity("b")], scope)
        found = await store.find_entities(query="a", scope=scope)
        assert {e.id for e in found} == {"a"}

    @pytest.mark.asyncio
    async def test_idempotent_upsert(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a")], scope)
        await store.upsert_entities([_entity("a", name="A2")], scope)
        result = await store.find_entities(query="a", scope=scope)
        assert len(result) == 1
        # Re-upsert wins on metadata
        assert result[0].name == "A2"


class TestFindEntities:
    @pytest.mark.asyncio
    async def test_exact_name_match_outranks_substring(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities(
            [
                _entity("alpha-corp", name="Alpha Corp"),
                _entity("alpha-prefix", name="Alpha Foundation"),
            ],
            scope,
        )
        result = await store.find_entities(query="Alpha Corp", scope=scope, k=2)
        # Exact name match comes first.
        assert result[0].id == "alpha-corp"

    @pytest.mark.asyncio
    async def test_type_filter(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities(
            [
                _entity("acme", type_="supplier", name="acme"),
                _entity("acme-product", type_="product", name="acme widget"),
            ],
            scope,
        )
        suppliers = await store.find_entities(query="acme", scope=scope, type_filter=["supplier"])
        assert {e.id for e in suppliers} == {"acme"}

    @pytest.mark.asyncio
    async def test_unknown_scope_returns_empty(self):
        store = InMemoryGraphStore()
        await store.upsert_entities([_entity("a")], _scope("-A"))
        result = await store.find_entities(query="a", scope=_scope("-B"))
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a")], scope)
        assert await store.find_entities(query="", scope=scope) == []
        assert await store.find_entities(query="   ", scope=scope) == []


class TestNeighbours:
    @pytest.mark.asyncio
    async def test_zero_hops_returns_seed_only(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a"), _entity("b")], scope)
        await store.upsert_edges([_edge("a", "b")], scope)
        ents, edges = await store.neighbours(["a"], scope=scope, max_hops=0)
        assert {e.id for e in ents} == {"a"}
        assert edges == []

    @pytest.mark.asyncio
    async def test_one_hop_includes_direct_neighbour(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a"), _entity("b"), _entity("c")], scope)
        await store.upsert_edges([_edge("a", "b"), _edge("b", "c")], scope)
        ents, edges = await store.neighbours(["a"], scope=scope, max_hops=1)
        assert {e.id for e in ents} == {"a", "b"}
        assert len(edges) == 1

    @pytest.mark.asyncio
    async def test_two_hops_reveals_z(self):
        """Query mentions A, walking 2 hops reveals Z."""
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities(
            [_entity("A"), _entity("B"), _entity("Z"), _entity("unrelated")],
            scope,
        )
        await store.upsert_edges(
            [_edge("A", "B", "supplies"), _edge("B", "Z", "owns")],
            scope,
        )
        ents, edges = await store.neighbours(["A"], scope=scope, max_hops=2)
        assert "Z" in {e.id for e in ents}
        assert len(edges) == 2

    @pytest.mark.asyncio
    async def test_undirected_traversal_follows_incoming_edges(self):
        """Walks should traverse both outgoing and incoming edges."""
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a"), _entity("b")], scope)
        await store.upsert_edges([_edge("b", "a", "manages")], scope)
        ents, _edges = await store.neighbours(["a"], scope=scope, max_hops=1)
        # Even though the edge is b → a, walking from a should still reach b.
        assert {e.id for e in ents} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_relation_filter(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a"), _entity("b"), _entity("c")], scope)
        await store.upsert_edges([_edge("a", "b", "supplies"), _edge("a", "c", "owns")], scope)
        ents, edges = await store.neighbours(["a"], scope=scope, max_hops=1, relation_filter=["supplies"])
        # "owns" edge filtered out — only b reachable.
        assert {e.id for e in ents} == {"a", "b"}
        assert all(e.relation == "supplies" for e in edges)

    @pytest.mark.asyncio
    async def test_scope_isolation(self):
        store = InMemoryGraphStore()
        await store.upsert_entities([_entity("a"), _entity("b")], _scope("-X"))
        await store.upsert_edges([_edge("a", "b")], _scope("-X"))
        # Different scope sees nothing.
        ents, edges = await store.neighbours(["a"], scope=_scope("-Y"))
        assert ents == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_unknown_seed_returns_empty(self):
        store = InMemoryGraphStore()
        scope = _scope()
        await store.upsert_entities([_entity("a")], scope)
        ents, edges = await store.neighbours(["does-not-exist"], scope=scope)
        assert ents == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_is_null_marker(self):
        assert InMemoryGraphStore.is_null is False
