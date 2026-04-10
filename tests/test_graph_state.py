"""Tests for src.graph.state — merge_dicts + replace_list reducers."""
from __future__ import annotations

from orchid.graph.state import merge_dicts, replace_list


# ── merge_dicts ─────────────────────────────────────────────────


class TestMergeDicts:
    def test_both_none_returns_empty_dict(self):
        assert merge_dicts(None, None) == {}

    def test_existing_plus_none_returns_existing(self):
        existing = {"a": 1, "b": 2}
        assert merge_dicts(existing, None) == existing

    def test_none_plus_new_returns_new(self):
        new = {"x": 10}
        assert merge_dicts(None, new) == new

    def test_shallow_merge_preserves_existing(self):
        existing = {"a": 1, "b": 2}
        new = {"c": 3}
        result = merge_dicts(existing, new)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_shallow_merge_updates_overlapping(self):
        existing = {"a": 1, "b": 2}
        new = {"b": 99, "c": 3}
        result = merge_dicts(existing, new)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_does_not_mutate_inputs(self):
        existing = {"a": 1}
        new = {"b": 2}
        merge_dicts(existing, new)
        assert existing == {"a": 1}
        assert new == {"b": 2}


# ── replace_list ────────────────────────────────────────────────


class TestReplaceList:
    def test_both_none_returns_empty_list(self):
        assert replace_list(None, None) == []

    def test_existing_plus_none_returns_existing(self):
        existing = ["a", "b"]
        assert replace_list(existing, None) == existing

    def test_none_plus_new_returns_new(self):
        new = ["x", "y"]
        assert replace_list(None, new) == new

    def test_new_replaces_existing(self):
        existing = ["a", "b"]
        new = ["c"]
        result = replace_list(existing, new)
        assert result == ["c"]

    def test_empty_new_replaces_existing(self):
        existing = ["a", "b"]
        result = replace_list(existing, [])
        assert result == []
