"""Tests for ``Neo4jGraphStore``.

The Neo4j extra ships behind ``pip install orchid-ai[neo4j]``.  The
default test environment doesn't include the driver, so the
deterministic test below covers only the missing-extra path.
Integration tests against a real Neo4j live in a separate gated
suite (NEO4J_URL env var).
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_missing_extra_raises_import_error():
    """Constructing ``Neo4jGraphStore`` without ``neo4j`` raises ImportError."""
    import orchid_ai.rag.backends.neo4j_graph as mod

    saved = sys.modules.get("neo4j")
    sys.modules["neo4j"] = None  # type: ignore[assignment]
    try:
        importlib.reload(mod)
        with pytest.raises(ImportError, match=r"pip install orchid-ai\[neo4j\]"):
            mod.Neo4jGraphStore()
    finally:
        if saved is None:
            sys.modules.pop("neo4j", None)
        else:
            sys.modules["neo4j"] = saved
        importlib.reload(mod)
