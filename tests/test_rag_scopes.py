"""Tests for src.rag.scopes — OrchidRAGScope + build_qdrant_filter."""

from __future__ import annotations

import dataclasses

import pytest
from qdrant_client.models import Filter

from orchid_ai.rag.scopes import SHARED_TENANT, OrchidRAGScope, build_qdrant_filter


# ── OrchidRAGScope dataclass ──────────────────────────────────────────


class TestRAGScope:
    def test_frozen(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.tenant_id = "t-2"  # type: ignore[misc]

    def test_defaults(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        assert scope.tenant_id == "t-1"
        assert scope.user_id == ""
        assert scope.chat_id == ""
        assert scope.agent_id == ""

    def test_stores_all_fields(self):
        scope = OrchidRAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1", agent_id="a-1")
        assert scope.tenant_id == "t-1"
        assert scope.user_id == "u-1"
        assert scope.chat_id == "c-1"
        assert scope.agent_id == "a-1"


# ── SHARED_TENANT constant ─────────────────────────────────────


def test_shared_tenant_constant():
    assert SHARED_TENANT == "__shared__"


# ── build_qdrant_filter ────────────────────────────────────────


class TestBuildQdrantFilter:
    def test_tenant_only_produces_2_clauses(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        f = build_qdrant_filter(scope)
        assert isinstance(f, Filter)
        assert f.should is not None
        assert len(f.should) == 2

    def test_tenant_plus_user_produces_3_clauses(self):
        scope = OrchidRAGScope(tenant_id="t-1", user_id="u-1")
        f = build_qdrant_filter(scope)
        assert len(f.should) == 3

    def test_tenant_user_chat_produces_4_clauses(self):
        scope = OrchidRAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1")
        f = build_qdrant_filter(scope)
        assert len(f.should) == 4

    def test_all_fields_produces_5_clauses(self):
        scope = OrchidRAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1", agent_id="a-1")
        f = build_qdrant_filter(scope)
        assert len(f.should) == 5

    def test_uses_should_not_must(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        f = build_qdrant_filter(scope)
        assert f.should is not None
        assert f.must is None

    def test_shared_clause_has_shared_tenant(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        f = build_qdrant_filter(scope)
        shared_clause = f.should[0]
        # The first clause's must list should contain a FieldCondition
        # matching tenant_id = "__shared__"
        assert any(
            getattr(cond, "key", None) == "tenant_id" and getattr(cond.match, "value", None) == SHARED_TENANT
            for cond in shared_clause.must
        )

    def test_tenant_clause_has_scope_tenant(self):
        scope = OrchidRAGScope(tenant_id="t-1")
        f = build_qdrant_filter(scope)
        tenant_clause = f.should[1]
        keys_and_values = {getattr(c, "key", None): getattr(c.match, "value", None) for c in tenant_clause.must}
        assert keys_and_values["tenant_id"] == "t-1"
        assert keys_and_values["scope"] == "tenant"

    def test_agent_id_without_user_and_chat_ignored(self):
        """agent_id alone (no user_id/chat_id) should NOT add extra clauses."""
        scope = OrchidRAGScope(tenant_id="t-1", agent_id="a-1")
        f = build_qdrant_filter(scope)
        assert len(f.should) == 2  # only shared + tenant
