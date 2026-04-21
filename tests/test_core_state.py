"""Tests for OrchidAuthContext and OrchidAgentState from src/core/state.py."""

from __future__ import annotations

import time


from orchid_ai.core.state import OrchidAgentState, OrchidAuthContext


# ── OrchidAuthContext defaults ──


def test_auth_defaults():
    ctx = OrchidAuthContext(access_token="tok")
    assert ctx.access_token == "tok"
    assert ctx.tenant_key == "default"
    assert ctx.user_id == ""
    assert ctx.expires_at == 0.0
    assert ctx.extra == {}


def test_auth_custom_values():
    ctx = OrchidAuthContext(
        access_token="tok",
        tenant_key="acme",
        user_id="u-1",
        expires_at=9999.0,
        extra={"role": "admin"},
    )
    assert ctx.tenant_key == "acme"
    assert ctx.user_id == "u-1"
    assert ctx.expires_at == 9999.0
    assert ctx.extra == {"role": "admin"}


def test_tenant_key_falls_back_to_default_on_empty_string():
    ctx = OrchidAuthContext(access_token="tok", tenant_key="")
    assert ctx.tenant_key == "default"


def test_user_id_returns_set_value():
    ctx = OrchidAuthContext(access_token="tok", user_id="u-42")
    assert ctx.user_id == "u-42"


# ── is_expired ──


def test_is_expired_false_when_zero():
    ctx = OrchidAuthContext(access_token="tok")
    assert ctx.is_expired is False


def test_is_expired_true_when_in_past():
    ctx = OrchidAuthContext(access_token="tok", expires_at=1.0)
    assert ctx.is_expired is True


def test_is_expired_false_when_far_future():
    ctx = OrchidAuthContext(access_token="tok", expires_at=time.time() + 99999)
    assert ctx.is_expired is False


# ── bearer_header ──


def test_bearer_header():
    ctx = OrchidAuthContext(access_token="my-token")
    assert ctx.bearer_header == {"Authorization": "Bearer my-token"}


# ── __repr__ ──


def test_repr_contains_key_info():
    ctx = OrchidAuthContext(access_token="tok", tenant_key="acme", user_id="u-1")
    r = repr(ctx)
    assert "OrchidAuthContext" in r
    assert "acme" in r
    assert "u-1" in r
    assert "expired=" in r


# ── __eq__ and __hash__ ──


def test_eq_same_values():
    a = OrchidAuthContext(access_token="tok", tenant_key="t", user_id="u")
    b = OrchidAuthContext(access_token="tok", tenant_key="t", user_id="u")
    assert a == b


def test_eq_different_values():
    a = OrchidAuthContext(access_token="tok1", tenant_key="t", user_id="u")
    b = OrchidAuthContext(access_token="tok2", tenant_key="t", user_id="u")
    assert a != b


def test_eq_not_implemented_for_non_auth():
    ctx = OrchidAuthContext(access_token="tok")
    assert ctx.__eq__("not an auth") is NotImplemented


def test_hash_matches_for_equal_instances():
    a = OrchidAuthContext(access_token="tok", tenant_key="t", user_id="u")
    b = OrchidAuthContext(access_token="tok", tenant_key="t", user_id="u")
    assert hash(a) == hash(b)


# ── extra is mutable ──


def test_extra_is_mutable():
    ctx = OrchidAuthContext(access_token="tok")
    ctx.extra["new_key"] = "value"
    assert ctx.extra["new_key"] == "value"


# ── Subclassing ──


def test_subclass_overrides_properties():
    class CustomAuth(OrchidAuthContext):
        def __init__(self, *, access_token, custom_tenant, custom_user, **kwargs):
            super().__init__(access_token=access_token, **kwargs)
            self._custom_tenant = custom_tenant
            self._custom_user = custom_user

        @property
        def tenant_key(self) -> str:
            return self._custom_tenant

        @property
        def user_id(self) -> str:
            return self._custom_user

    ctx = CustomAuth(access_token="tok", custom_tenant="my-tenant", custom_user="my-user")
    assert ctx.tenant_key == "my-tenant"
    assert ctx.user_id == "my-user"
    assert ctx.access_token == "tok"


# ── OrchidAgentState ──


def test_agent_state_empty_dict():
    state: OrchidAgentState = {}
    assert isinstance(state, dict)


def test_agent_state_with_fields():
    state: OrchidAgentState = {
        "messages": [],
        "auth_context": OrchidAuthContext(access_token="tok"),
        "chat_id": "c-1",
        "active_agents": ["a"],
        "mcp_context": {},
        "rag_context": {},
        "final_response": None,
        "skill_instructions": {},
    }
    assert state["chat_id"] == "c-1"
    assert state["final_response"] is None


def test_agent_state_is_total_false():
    # total=False means __required_keys__ is empty
    assert OrchidAgentState.__required_keys__ == frozenset()
