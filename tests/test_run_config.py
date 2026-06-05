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


def test_node_config_param_is_langgraph_injectable():
    """Node `config` params must use an annotation LangGraph accepts, else it
    silently does NOT inject config and auth_from_config(config) sees None.

    Regression for the `RunnableConfig | None` (PEP 604 union string under
    `from __future__ import annotations`) that LangGraph does not match.
    """
    import warnings

    from orchid_ai import OrchidRuntime
    from orchid_ai.config.schema import (
        OrchidAgentConfig,
        OrchidAgentsConfig,
        OrchidLLMConfig,
        OrchidRAGConfig,
    )
    from orchid_ai.graph.graph import build_graph

    cfg = OrchidAgentsConfig(
        agents={
            "a": OrchidAgentConfig(
                description="d",
                prompt="p",
                rag=OrchidRAGConfig(enabled=False),
                llm=OrchidLLMConfig(model="ollama/llama3.2"),
            )
        }
    )
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_graph(config=cfg, runtime=runtime)
    typing_warnings = [str(w.message) for w in caught if "config' parameter should be typed" in str(w.message)]
    assert typing_warnings == [], typing_warnings
