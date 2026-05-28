"""Tests for the friendly ImportError gates on optional document parsers.

The ``[documents]`` extra is now opt-in (H3 from the 2026-05-28 code
review).  When a user runs a lean ``pip install orchid-ai`` and tries
to parse a PDF / DOCX / XLSX, we want:

1. Importing :mod:`orchid_ai.documents.parsers` to keep working — no
   eager third-party imports at module level.
2. Calling ``parse()`` on a heavy parser to raise a clean
   :class:`ImportError` mentioning ``pip install orchid-ai[documents]``
   so the user knows exactly what to install.

We simulate "the package isn't installed" by stuffing a sentinel into
``sys.modules`` so the lazy ``import fitz`` / ``from docx import ...`` /
``from openpyxl import ...`` line raises ``ImportError`` even though
the real package is installed in the test venv.  This mirrors the
behaviour an end user would see on a clean install.
"""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager

import pytest

from orchid_ai.documents.parsers import (
    CSVParser,
    DOCXParser,
    PDFParser,
    TextParser,
    XLSXParser,
    _DOCUMENTS_EXTRA_HINT,
    _missing_extra,
)


# ── _missing_extra builder ─────────────────────────────────────────


def test_missing_extra_message_shape():
    """The helper produces a uniform, actionable error message."""
    err = _missing_extra("PDF", "fitz", "pymupdf")
    assert isinstance(err, ImportError)
    msg = str(err)
    assert "PDF parsing requires" in msg
    assert "fitz" in msg
    assert "pymupdf" in msg
    assert "pip install orchid-ai[documents]" in msg


def test_documents_extra_hint_mentions_pip():
    """The hint string is the single source of truth for the
    actionable command — keep it specific."""
    assert "pip install orchid-ai[documents]" in _DOCUMENTS_EXTRA_HINT


# ── Lean-install simulation helpers ────────────────────────────────


@contextmanager
def _block_imports(*module_prefixes: str):
    """Temporarily make ``import <prefix>`` raise ``ImportError``.

    Patches :func:`builtins.__import__` and clears any cached
    submodules so the lazy import inside the parser hits the gate.
    """
    cached = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if any(name == p or name.startswith(f"{p}.") for p in module_prefixes)
    }
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == p or name.startswith(f"{p}.") for p in module_prefixes):
            raise ImportError(f"No module named {name!r} (simulated)")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(cached)


# ── Per-parser ImportError surface ─────────────────────────────────


@pytest.mark.asyncio
async def test_pdf_parser_raises_friendly_import_error():
    parser = PDFParser()
    with _block_imports("fitz"):
        with pytest.raises(ImportError) as excinfo:
            await parser.parse(b"%PDF-1.4 minimal stub", "doc.pdf")
    msg = str(excinfo.value)
    assert "PDF parsing requires" in msg
    assert "pymupdf" in msg
    assert "pip install orchid-ai[documents]" in msg


@pytest.mark.asyncio
async def test_docx_parser_raises_friendly_import_error():
    parser = DOCXParser()
    with _block_imports("docx"):
        with pytest.raises(ImportError) as excinfo:
            await parser.parse(b"PK fake docx", "doc.docx")
    msg = str(excinfo.value)
    assert "DOCX parsing requires" in msg
    assert "python-docx" in msg
    assert "pip install orchid-ai[documents]" in msg


@pytest.mark.asyncio
async def test_xlsx_parser_raises_friendly_import_error():
    parser = XLSXParser()
    with _block_imports("openpyxl"):
        with pytest.raises(ImportError) as excinfo:
            await parser.parse(b"PK fake xlsx", "sheet.xlsx")
    msg = str(excinfo.value)
    assert "XLSX parsing requires" in msg
    assert "openpyxl" in msg
    assert "pip install orchid-ai[documents]" in msg


# ── Parsers that don't require the extras ──────────────────────────


@pytest.mark.asyncio
async def test_csv_parser_works_without_extras():
    """:class:`CSVParser` only uses stdlib — no extras gate."""
    parser = CSVParser()
    out = await parser.parse(b"a,b,c\n1,2,3\n", "data.csv")
    assert "a | b | c" in out
    assert "1 | 2 | 3" in out


@pytest.mark.asyncio
async def test_text_parser_works_without_extras():
    """:class:`TextParser` only uses stdlib — no extras gate."""
    parser = TextParser()
    out = await parser.parse("hello world".encode("utf-8"), "note.md")
    assert out == "hello world"


# ── Module imports remain cheap ────────────────────────────────────


def test_parsers_module_does_not_import_heavy_packages_eagerly():
    """Importing :mod:`orchid_ai.documents.parsers` must not pull in
    ``fitz`` / ``docx`` / ``openpyxl`` / ``PIL`` at module load time
    — otherwise a lean ``pip install orchid-ai`` user crashes on
    `from orchid_ai.documents.parsers import CSVParser`.

    We can't unimport packages already loaded by other tests, so we
    inspect the parser module's own AST via ``ast.parse`` and assert
    that no top-level ``import`` / ``from … import`` statement
    references one of the heavy packages.
    """
    import ast
    import inspect

    import orchid_ai.documents.parsers as parsers

    source = inspect.getsource(parsers)
    tree = ast.parse(source)
    heavy = {"fitz", "docx", "openpyxl", "PIL"}

    top_level_imports: set[str] = set()
    for node in tree.body:  # only top-level — function bodies are fine
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_imports.add(node.module.split(".", 1)[0])

    assert not (heavy & top_level_imports), (
        f"Heavy packages must not be imported at module level — found: {sorted(heavy & top_level_imports)}"
    )
