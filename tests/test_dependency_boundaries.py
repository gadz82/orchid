"""Architectural boundary lint — concrete-backend imports stay inside ``rag/backends/``.

No module under ``orchid_ai/`` may import a concrete vector DB / graph
DB / sparse-encoder client outside ``rag/backends/<name>.py``.  This
test walks the package's ``ast`` and asserts the rule.

The list of forbidden modules grows as we add backends — keep the
``ALLOWED_LOCATIONS`` table accurate when introducing a new client.

Two distinct rule shapes live here:

* :data:`ALLOWED_LOCATIONS` — ``module → list of allowed *file or
  directory* paths``.  Empty list means "nowhere" (the import is
  banned everywhere in the package).  Directory entries match
  recursively so we don't have to re-list every file under, say,
  ``rag/`` whenever a new module appears.
* :data:`CORE_FORBIDDEN` — modules that must not appear under
  ``core/``.  This is the H1 rule from the 2026-05-28 review:
  ``core/`` ships no LangChain dependency, so it must not import
  ``langchain`` or ``langchain_core`` anywhere — even though those
  modules are perfectly fine elsewhere in the package.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "orchid_ai"


# Each entry maps a forbidden top-level module name to a list of
# *allowed* relative paths under PACKAGE_ROOT.  An entry can be either:
#
# * a Python file (e.g. ``"rag/backends/qdrant.py"``) — exact match.
# * a directory (e.g. ``"rag/backends/"``) — matches every ``*.py``
#   below that directory recursively.
#
# An empty list means "this import is forbidden everywhere in the
# package" — the safest default for plugin-only backends that no
# longer ship in-tree.
#
# Note: ``qdrant_client`` and ``neo4j`` moved to plugin packages
# (orchid-rag-qdrant, orchid-rag-neo4j) and are no longer present in
# the core library.
#
# ``langchain`` (the umbrella package) is forbidden everywhere — the
# framework intentionally builds against ``langchain_core`` only and
# delegates anything heavier (chains, agents, vector-stores) to
# integrators / plugins.  ``langchain_core`` itself is allowed
# throughout the package *except* under ``core/`` — see
# :data:`CORE_FORBIDDEN` below for that finer-grained rule.
ALLOWED_LOCATIONS: dict[str, list[str]] = {
    "opensearchpy": [],
    "weaviate": [],
    "weaviate_client": [],
    "memgraph": [],
    "psycopg": [],
    "psycopg2": [],
    "aiosqlite": [],
    "langchain": [],
}


# Import strings that must never appear anywhere under ``orchid_ai/``.
# Checked with a text scan (not an import scan) so plugin/backend
# modules cannot sneak back into the core library.
FORBIDDEN_SOURCE_STRINGS: list[str] = [
    "langgraph.checkpoint.sqlite",
    "langgraph.checkpoint.postgres",
]


# Modules that must never appear under ``core/``.  ``core/`` is the
# zero-third-party-dependency layer; the framework's canonical document
# model is :class:`orchid_ai.core.repository.OrchidDocument` (a stdlib
# dataclass), and :mod:`orchid_ai.rag.adapters` provides the thin
# ``OrchidDocument`` ↔ LangChain ``Document`` conversion layer used by
# RAG backends.  Any LangChain import sneaking back into ``core/``
# violates the H1 architectural rule from the 2026-05-28 code review.
#
# ``httpx`` and ``pydantic`` are listed for the same reason: decision
# adapters and configuration models live outside ``core/``, so the
# provider contract stays transport- and schema-free.
CORE_FORBIDDEN: list[str] = ["langchain", "langchain_core", "httpx", "pydantic"]


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
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:  # absolute import only
            names.add(node.module.split(".", 1)[0])
    return names


def _is_allowed(file_path: Path, allowed_entries: list[str]) -> bool:
    """Return ``True`` when ``file_path`` matches any allowed file or directory."""
    file_resolved = file_path.resolve()
    for entry in allowed_entries:
        candidate = (PACKAGE_ROOT / entry).resolve()
        if entry.endswith(".py"):
            if file_resolved == candidate:
                return True
        else:
            try:
                file_resolved.relative_to(candidate)
                return True
            except ValueError:
                continue
    return False


@pytest.mark.parametrize("forbidden", sorted(ALLOWED_LOCATIONS))
def test_concrete_backend_clients_are_isolated(forbidden: str) -> None:
    allowed_entries = ALLOWED_LOCATIONS[forbidden]

    violations: list[str] = []
    for py_file in _python_files():
        if forbidden not in _imported_names(py_file):
            continue
        if not _is_allowed(py_file, allowed_entries):
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))

    assert not violations, (
        f"Concrete backend client {forbidden!r} imported outside its allowed location "
        f"({allowed_entries!r}):\n  - " + "\n  - ".join(violations)
    )


@pytest.mark.parametrize("forbidden", CORE_FORBIDDEN)
def test_core_has_no_forbidden_imports(forbidden: str) -> None:
    """``core/`` must not import any concrete backend or framework library.

    The framework's canonical document model is
    :class:`orchid_ai.core.repository.OrchidDocument`, a stdlib
    dataclass.  Anything that needs to interoperate with LangChain
    goes through :mod:`orchid_ai.rag.adapters`.
    """
    core_dir = PACKAGE_ROOT / "core"
    violations: list[str] = []
    for py_file in core_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        if forbidden in _imported_names(py_file):
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))

    assert not violations, (
        f"{forbidden!r} imported inside core/ — core/ must remain dependency-free.\n"
        f"Move the concrete implementation outside core/ and depend on an ABC instead.\n  - "
        + "\n  - ".join(violations)
    )


def test_no_legacy_null_module() -> None:
    """``rag/null.py`` was removed in favour of ``rag/backends/null.py``."""
    legacy = PACKAGE_ROOT / "rag" / "null.py"
    assert not legacy.exists(), f"{legacy} should be deleted (use rag/backends/null.py)"


@pytest.mark.parametrize("forbidden", FORBIDDEN_SOURCE_STRINGS)
def test_concrete_checkpoint_backends_are_not_referenced(forbidden: str) -> None:
    """Durable checkpoint backends live in storage plugins.

    A text scan (not an import scan) so no module — not even a docstring
    or a lazy import inside a function — can reintroduce the dependency.
    """
    violations: list[str] = []
    for py_file in _python_files():
        if forbidden in py_file.read_text(encoding="utf-8"):
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))

    assert not violations, (
        f"{forbidden!r} referenced inside orchid_ai/ — durable checkpoint backends "
        f"belong in the storage plugin packages.\n  - " + "\n  - ".join(violations)
    )


def test_decision_contract_imports_only_stdlib() -> None:
    """``core/decision.py`` is the decision port; it must stay stdlib-only.

    The denylist above catches known offenders. This test proves the stronger
    invariant for the new contract: every top-level import is in the standard
    library (or the package itself).
    """
    module = PACKAGE_ROOT / "core" / "decision.py"
    imported = _imported_names(module)
    non_stdlib = {name for name in imported if name not in sys.stdlib_module_names and name != "orchid_ai"}
    assert not non_stdlib, f"core/decision.py imports non-stdlib modules: {sorted(non_stdlib)}"


def test_decision_adapters_are_not_reexported() -> None:
    """Adapters live in ``orchid_ai.decision``; the top level exposes the port only."""
    import orchid_ai

    for name in (
        "StateQuestionsDecisionModel",
        "OpenAICompatibleDecisionModel",
        "LLMDecisionModel",
        "StubDecisionModel",
        "DecisionHttpClient",
    ):
        assert not hasattr(orchid_ai, name), f"{name} must not be re-exported from orchid_ai"


def test_decision_adapters_import_no_vendor_sdk() -> None:
    """The adapters speak HTTP dialects directly; no provider SDK is imported."""
    decision_dir = PACKAGE_ROOT / "decision"
    forbidden = {"litellm", "openai", "anthropic", "cohere", "groq", "mistralai"}
    violations: list[str] = []
    for py_file in decision_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        imported = _imported_names(py_file)
        bad = imported & forbidden
        if bad:
            violations.append(f"{py_file.relative_to(PACKAGE_ROOT)} imports {sorted(bad)}")
    assert not violations, "vendor SDK imported in decision/: " + "; ".join(violations)
