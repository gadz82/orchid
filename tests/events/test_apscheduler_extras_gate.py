"""Tests for the friendly ImportError gate on the APScheduler backend.

The ``[events]`` extra is opt-in (H3 from the 2026-05-28 code review).
When a user runs a lean ``pip install orchid-ai`` and tries to
construct :class:`APSchedulerBackend`, we want:

1. Importing :mod:`orchid_ai.events.schedulers.apscheduler` to keep
   working — the heavy :mod:`apscheduler` import is gated behind
   ``_apscheduler_imports()``.
2. Constructing the backend (or calling any registration method) to
   raise a clean :class:`ImportError` mentioning
   ``pip install orchid-ai[events]``.

We simulate "the package isn't installed" by stuffing a sentinel
into :data:`sys.modules` so the lazy ``from apscheduler... import``
statement raises ``ImportError`` even though the real package is
installed in the test venv.  This mirrors the behaviour an end user
would see on a clean install.
"""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager

import pytest

import orchid_ai.events.schedulers.apscheduler as apscheduler_mod
from orchid_ai.events.schedulers.apscheduler import APSchedulerBackend, _apscheduler_imports


@contextmanager
def _block_apscheduler_import():
    """Temporarily make ``import apscheduler[.*]`` raise ``ImportError``."""
    cached = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "apscheduler" or name.startswith("apscheduler.")
    }
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "apscheduler" or name.startswith("apscheduler."):
            raise ImportError(f"No module named {name!r} (simulated)")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(cached)


def test_apscheduler_imports_raises_friendly_import_error():
    """``_apscheduler_imports`` raises :class:`ImportError` (not
    ``RuntimeError``) so callers can ``except ImportError`` uniformly
    across the parsers + scheduler extras gates."""
    with _block_apscheduler_import(), pytest.raises(ImportError) as excinfo:
        _apscheduler_imports()
    msg = str(excinfo.value)
    assert "APSchedulerBackend requires" in msg
    assert "apscheduler" in msg
    assert "pip install orchid-ai[events]" in msg


def test_apscheduler_backend_constructor_raises_friendly_import_error():
    """Constructing :class:`APSchedulerBackend` triggers the gate via
    its call to ``_apscheduler_imports()`` — so a lean install user
    sees the same ImportError directly from the public constructor."""
    with _block_apscheduler_import(), pytest.raises(ImportError) as excinfo:
        APSchedulerBackend()
    msg = str(excinfo.value)
    assert "pip install orchid-ai[events]" in msg


def test_apscheduler_module_does_not_import_apscheduler_eagerly():
    """Importing :mod:`orchid_ai.events.schedulers.apscheduler` must
    not pull in ``apscheduler`` at module load time — every reference
    must live behind ``_apscheduler_imports()`` or inside a method
    body that runs only when the scheduler is in use."""
    import ast
    import inspect

    source = inspect.getsource(apscheduler_mod)
    tree = ast.parse(source)

    top_level_imports: set[str] = set()
    for node in tree.body:  # only top-level — function bodies are fine
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", 1)[0])

    assert "apscheduler" not in top_level_imports, (
        "apscheduler must not be imported at module level — gate every "
        "import behind ``_apscheduler_imports()`` or a method body so "
        "the [events] extra stays optional."
    )
