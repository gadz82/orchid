"""Architectural boundary lint — concrete-backend imports stay inside ``rag/backends/``.

ADR-028 says no module under ``orchid_ai/`` may import a concrete vector
DB / graph DB / sparse-encoder client outside ``rag/backends/<name>.py``.
This test walks the package's ``ast`` and asserts the rule.

The list of forbidden modules grows as we add backends — keep the
``ALLOWED_LOCATIONS`` table accurate when introducing a new client.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "orchid_ai"


# Each entry maps a forbidden top-level module name to the relative path
# (under PACKAGE_ROOT) where it is *allowed* to be imported.  Empty list
# means "nowhere" — current state for backends we haven't shipped yet.
ALLOWED_LOCATIONS: dict[str, list[str]] = {
    "qdrant_client": [
        "rag/backends/qdrant.py",
        "rag/backends/qdrant_doc_store.py",
    ],
    "neo4j": ["rag/backends/neo4j_graph.py"],
    "opensearchpy": [],
    "weaviate": [],
    "weaviate_client": [],
    "memgraph": [],
    "psycopg": [],
    "psycopg2": [],
}


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_names(path: Path) -> set[str]:
    """Return the set of top-level package names imported by ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover — surfaces clearly when it happens
        raise AssertionError(f"Syntax error in {path}: {exc}") from exc

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:  # absolute import only
                names.add(node.module.split(".", 1)[0])
    return names


@pytest.mark.parametrize("forbidden", sorted(ALLOWED_LOCATIONS))
def test_concrete_backend_clients_are_isolated(forbidden: str) -> None:
    allowed = {Path(p).resolve() for p in (PACKAGE_ROOT / loc for loc in ALLOWED_LOCATIONS[forbidden])}

    violations: list[str] = []
    for py_file in _python_files():
        if forbidden not in _imported_names(py_file):
            continue
        if py_file.resolve() not in allowed:
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))

    assert not violations, (
        f"Concrete backend client {forbidden!r} imported outside its allowed location "
        f"({ALLOWED_LOCATIONS[forbidden]!r}):\n  - " + "\n  - ".join(violations)
    )


def test_no_legacy_null_module() -> None:
    """``rag/null.py`` was removed in favour of ``rag/backends/null.py``."""
    legacy = PACKAGE_ROOT / "rag" / "null.py"
    assert not legacy.exists(), f"{legacy} should be deleted (use rag/backends/null.py)"
