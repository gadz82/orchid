"""When ``events`` is omitted (or disabled) nothing wakes up.

This is a *floor* test rather than an exhaustive one.  It asserts:

1. An agents-config without an ``events:`` block leaves
   ``OrchidAgentsConfig.events`` as ``None`` so the lifespan code can
   short-circuit on a single ``if cfg.events and cfg.events.enabled``
   check.
2. A config with ``events.enabled: false`` parses and leaves the
   infra fields empty (placeholder block).
3. The ``orchid_ai/events/__init__.py`` performs **no class
   instantiation at module-load time** — verified by AST.  The
   actual lifespan integration that consumes this contract lands in
   the API phase; this test guards the framework half of the bargain.
"""

from __future__ import annotations

import ast
from pathlib import Path

from orchid_ai.config.schema import OrchidAgentsConfig

_EVENTS_INIT = Path(__file__).resolve().parents[2] / "orchid_ai" / "events" / "__init__.py"


def test_agents_config_without_events_block_leaves_events_none() -> None:
    cfg = OrchidAgentsConfig.model_validate({"version": "1"})
    assert cfg.events is None


def test_agents_config_with_disabled_events_block_parses() -> None:
    cfg = OrchidAgentsConfig.model_validate({"version": "1", "events": {"enabled": False}})
    assert cfg.events is not None
    assert cfg.events.enabled is False
    # Disabled blocks must NOT carry a fully-wired infra config — they
    # are placeholders.
    assert cfg.events.store is None
    assert cfg.events.queue is None


def test_events_init_has_no_module_level_class_instantiation() -> None:
    """Walk the events package ``__init__.py`` AST and assert no
    ``Call`` nodes sit at module scope.

    Module-level imports / assignments are fine; calling something
    (``InMemorySignalQueue()`` or similar) at import time would mean
    the package wakes up infrastructure even when ``events.enabled``
    is ``False``."""
    tree = ast.parse(_EVENTS_INIT.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in tree.body:
        # Walk only top-level statements — nested defs / classes are
        # fine because their bodies don't run at import time.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Bare module docstring expression — fine.
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Assignments at module level are allowed iff the right
            # side has no Call.  A simple ``__all__ = [...]`` is fine;
            # ``_singleton = MyClass()`` is not.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    offenders.append(ast.unparse(node))
                    break
            continue
        offenders.append(ast.unparse(node))
    assert not offenders, (
        f"orchid_ai/events/__init__.py must not run code at import time. Offending statements: {offenders}"
    )
