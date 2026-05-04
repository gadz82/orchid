"""Spec §16 case 16 — shadow keys never collide.

Three minis with unique ``mini_id`` produce exactly three entries in
``mini_agent_outcomes`` keyed ``support#mini_0/1/2``.  The shallow
``merge_dicts`` reducer on ``mini_agent_outcomes`` and ``mcp_context``
does the right thing under parallel writes.
"""

from __future__ import annotations

from orchid_ai.graph.state import GraphState, merge_dicts


def test_merge_dicts_disjoint_keys_combine():
    """Three parallel writers with disjoint keys → three entries."""
    out_0 = {"support#mini_0": {"status": "ok", "summary": "A"}}
    out_1 = {"support#mini_1": {"status": "ok", "summary": "B"}}
    out_2 = {"support#mini_2": {"status": "ok", "summary": "C"}}

    after_0_1 = merge_dicts(out_0, out_1)
    after_0_1_2 = merge_dicts(after_0_1, out_2)

    assert set(after_0_1_2.keys()) == {
        "support#mini_0",
        "support#mini_1",
        "support#mini_2",
    }
    assert after_0_1_2["support#mini_0"]["summary"] == "A"
    assert after_0_1_2["support#mini_1"]["summary"] == "B"
    assert after_0_1_2["support#mini_2"]["summary"] == "C"


def test_merge_dicts_preserves_keys_from_other_parents():
    """Outcomes from other parents must coexist in the same channel."""
    a = {"support#mini_0": {"summary": "S"}}
    b = {"finance#mini_0": {"summary": "F"}}
    merged = merge_dicts(a, b)
    assert merged == {
        "support#mini_0": {"summary": "S"},
        "finance#mini_0": {"summary": "F"},
    }


def test_merge_dicts_handles_none_inputs():
    """None inputs (initial state) merge cleanly."""
    assert merge_dicts(None, {"k": "v"}) == {"k": "v"}
    assert merge_dicts({"k": "v"}, None) == {"k": "v"}
    assert merge_dicts(None, None) == {}


def test_state_typeddict_includes_mini_agent_channels():
    """The new fields are declared on ``GraphState``."""
    annotations = GraphState.__annotations__
    assert "mini_agent_outcomes" in annotations
    assert "mini_agent_decisions" in annotations
    # Sentinel keys for Send payloads — also declared.
    assert "_active_mini_parent" in annotations
    assert "_active_mini_id" in annotations
    assert "_active_mini_subtask" in annotations
    assert "_active_mini_tool_subset" in annotations
