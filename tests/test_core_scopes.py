"""Tests for scope-level resolution and scope-key derivation in core/scopes.py."""

from __future__ import annotations

from orchid_ai.core.scopes import (
    SHARED_TENANT,
    OrchidRAGScope,
    resolve_scope_level,
    scope_key,
)


class TestResolveScopeLevel:
    def test_tenant_level(self):
        scope = OrchidRAGScope(tenant_id="t1")
        assert resolve_scope_level(scope) == "tenant"

    def test_user_level(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1")
        assert resolve_scope_level(scope) == "user"

    def test_chat_shared_level(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1")
        assert resolve_scope_level(scope) == "chat_shared"

    def test_chat_agent_level(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")
        assert resolve_scope_level(scope) == "chat_agent"

    def test_shared_tenant_is_still_tenant_level(self):
        scope = OrchidRAGScope(tenant_id=SHARED_TENANT)
        assert resolve_scope_level(scope) == "tenant"


class TestScopeKey:
    def test_key_encodes_all_fields(self):
        scope = OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")
        assert scope_key(scope) == "t1\x1fu1\x1fc1\x1fa1"

    def test_different_tenant_produces_different_key(self):
        a = OrchidRAGScope(tenant_id="t1")
        b = OrchidRAGScope(tenant_id="t2")
        assert scope_key(a) != scope_key(b)

    def test_shared_vs_tenant_produces_different_key(self):
        shared = OrchidRAGScope(tenant_id=SHARED_TENANT)
        tenant = OrchidRAGScope(tenant_id="default")
        assert scope_key(shared) != scope_key(tenant)

    def test_user_vs_tenant_produces_different_key(self):
        user = OrchidRAGScope(tenant_id="t1", user_id="u1")
        tenant = OrchidRAGScope(tenant_id="t1")
        assert scope_key(user) != scope_key(tenant)

    def test_same_scope_produces_same_key(self):
        a = OrchidRAGScope(tenant_id="t1", user_id="u1")
        b = OrchidRAGScope(tenant_id="t1", user_id="u1")
        assert scope_key(a) == scope_key(b)
