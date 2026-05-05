"""Tests for ``SpladeEncoder``.

The Splade encoder is opt-in via ``pip install orchid-ai[splade]``.
The default test environment doesn't ship ``torch`` / ``transformers``,
so we only test the missing-extra error path here.  Integration tests
against a real Splade model live in a separate gated suite.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_missing_extra_raises_import_error():
    """Constructing ``SpladeEncoder`` without the extra raises ImportError
    with the install hint."""
    # Pretend torch / transformers aren't installed even if they happen
    # to be present in the test environment.
    import orchid_ai.rag.sparse.splade as splade_mod

    saved_torch = sys.modules.get("torch")
    saved_tx = sys.modules.get("transformers")
    sys.modules["torch"] = None  # type: ignore[assignment]
    sys.modules["transformers"] = None  # type: ignore[assignment]
    try:
        importlib.reload(splade_mod)
        with pytest.raises(ImportError, match=r"pip install orchid-ai\[splade\]"):
            splade_mod.SpladeEncoder()
    finally:
        # Restore real modules so other tests aren't poisoned.
        if saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch
        if saved_tx is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tx
        importlib.reload(splade_mod)
