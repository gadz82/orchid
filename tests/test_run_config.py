"""Tests for orchid_ai.core.run_config — auth as RunnableConfig context."""

from __future__ import annotations

from orchid_ai.core.run_config import CONFIG_KEY_AUTH, auth_from_config, with_auth
from orchid_ai.core.state import OrchidAuthContext


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="tok", tenant_key="acme", user_id="u1")


# ── with_auth ───────────────────────────────────────────────


def test_with_auth_puts_auth_under_configurable():
    auth = _auth()
    config = with_auth(auth)
    assert config["configurable"][CONFIG_KEY_AUTH] is auth


def test_with_auth_sets_thread_id():
    config = with_auth(_auth(), thread_id="chat-1")
    assert config["configurable"]["thread_id"] == "chat-1"


def test_with_auth_merges_base_without_mutating():
    base = {"configurable": {"request_id": "r-1"}, "callbacks": ["cb"]}
    config = with_auth(_auth(), thread_id="c-1", base=base)
    assert config["configurable"]["request_id"] == "r-1"
    assert config["configurable"]["thread_id"] == "c-1"
    assert config["callbacks"] == ["cb"]
    # base untouched
    assert "thread_id" not in base["configurable"]
    assert CONFIG_KEY_AUTH not in base["configurable"]


def test_with_auth_none_auth_is_not_written():
    config = with_auth(None, thread_id="c-1")
    assert CONFIG_KEY_AUTH not in config["configurable"]
    assert config["configurable"]["thread_id"] == "c-1"


# ── auth_from_config ────────────────────────────────────────


def test_auth_from_config_roundtrip():
    auth = _auth()
    assert auth_from_config(with_auth(auth)) is auth


def test_auth_from_config_none_safe():
    assert auth_from_config(None) is None
    assert auth_from_config({}) is None
    assert auth_from_config({"configurable": {}}) is None


# ── architecture guard ──────────────────────────────────────


def test_state_schemas_have_no_auth_channel():
    """auth must NOT be a graph-state channel (it lives in the config)."""
    from orchid_ai.core.state import OrchidAgentState
    from orchid_ai.graph.state import GraphState

    assert "auth_context" not in OrchidAgentState.__annotations__
    assert "auth_context" not in GraphState.__annotations__
