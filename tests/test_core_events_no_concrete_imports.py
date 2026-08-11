"""Boundary lint: ``orchid_ai/core/events/`` must not import from
``orchid_ai/events/`` (or any other concrete-side module).

Every other rule in this codebase is enforced by reading code; this
one we automate because the cost of an accidental concrete import in
``core/`` is high — it propagates to every consumer's transitive deps
and breaks the 'core has no external dependencies' contract.

Walks the AST.  Allowed top-level imports inside ``core/events/``:
stdlib + ``langchain_core`` (already permitted by the rest of
``core/``) + ``orchid_ai.core.*`` (intra-core).  Anything else fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_EVENTS = Path(__file__).resolve().parents[1] / "orchid_ai" / "core" / "events"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "orchid_ai"


_ALLOWED_TOP_LEVEL = {
    "__future__",
    "abc",
    "asyncio",
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "logging",
    "typing",
    "uuid",
    "langchain_core",  # consistent with the rest of core/
    "orchid_ai",  # intra-package — restricted to core.* below
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.append(node.module)
    return out


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _python_files(CORE_EVENTS), ids=lambda p: p.name)
def test_core_events_imports_are_pure(path: Path) -> None:
    for imp in _imports(path):
        top = imp.split(".", 1)[0]
        if top not in _ALLOWED_TOP_LEVEL:
            pytest.fail(f"{path.relative_to(PACKAGE_ROOT)} imports forbidden module {imp!r}")
        if top == "orchid_ai":
            # intra-package — must stay inside ``orchid_ai.core``
            assert imp.startswith("orchid_ai.core"), (
                f"{path.relative_to(PACKAGE_ROOT)} imports {imp!r} — "
                f"core/events/ may not reach into orchid_ai.events / config / etc."
            )


def test_no_events_concrete_module_imports_into_core() -> None:
    """No file under ``orchid_ai/core/`` (anywhere) should import from
    ``orchid_ai.events.`` — that would invert the dependency."""
    core_root = PACKAGE_ROOT / "core"
    violations: list[str] = []
    for path in _python_files(core_root):
        for imp in _imports(path):
            if imp.startswith("orchid_ai.events"):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {imp}")
    assert not violations, "\n".join(violations)
