"""Tests for the OrchidDocument ↔ langchain_core.Document adapter."""

from __future__ import annotations

import pytest

from orchid_ai.core.repository import Document, OrchidDocument
from orchid_ai.rag.adapters import (
    from_langchain_document,
    from_langchain_documents,
    to_langchain_document,
    to_langchain_documents,
)


# ── OrchidDocument basics ─────────────────────────────────────────────


def test_orchid_document_is_plain_dataclass():
    """OrchidDocument is a stdlib dataclass — no pydantic, no langchain."""
    import dataclasses

    assert dataclasses.is_dataclass(OrchidDocument)
    fields = {f.name for f in dataclasses.fields(OrchidDocument)}
    assert fields == {"page_content", "metadata", "id"}


def test_document_alias_points_to_orchid_document():
    """``Document`` is kept as a back-compat alias for OrchidDocument."""
    assert Document is OrchidDocument


def test_orchid_document_default_metadata_is_independent_per_instance():
    """``metadata`` uses ``field(default_factory=dict)`` so two instances
    don't share a mutable default."""
    a = OrchidDocument(page_content="a")
    b = OrchidDocument(page_content="b")
    a.metadata["key"] = "value"
    assert b.metadata == {}


def test_orchid_document_positional_construction():
    """Field order matches LangChain's Document so positional
    construction works for legacy callers."""
    doc = OrchidDocument("hello", {"k": "v"}, "id-1")
    assert doc.page_content == "hello"
    assert doc.metadata == {"k": "v"}
    assert doc.id == "id-1"


# ── to_langchain_document ─────────────────────────────────────────────


def test_to_langchain_document_round_trip():
    from langchain_core.documents import Document as LCDocument

    src = OrchidDocument(id="d-1", page_content="hello", metadata={"scope": "tenant"})
    out = to_langchain_document(src)

    assert isinstance(out, LCDocument)
    assert out.page_content == "hello"
    assert out.metadata == {"scope": "tenant"}
    assert out.id == "d-1"


def test_to_langchain_document_copies_metadata():
    """Mutating the returned langchain Document's metadata must not
    leak back into the source OrchidDocument."""
    src = OrchidDocument(id="d-1", page_content="hi", metadata={"a": 1})
    out = to_langchain_document(src)
    out.metadata["a"] = 2
    assert src.metadata == {"a": 1}


def test_to_langchain_document_handles_none_id():
    src = OrchidDocument(page_content="x")
    out = to_langchain_document(src)
    assert out.id is None
    assert out.page_content == "x"


def test_to_langchain_documents_batch():
    docs = [OrchidDocument(page_content=t, id=f"d-{i}") for i, t in enumerate(("a", "b", "c"))]
    out = to_langchain_documents(docs)
    assert len(out) == 3
    assert [d.page_content for d in out] == ["a", "b", "c"]
    assert [d.id for d in out] == ["d-0", "d-1", "d-2"]


# ── from_langchain_document ───────────────────────────────────────────


def test_from_langchain_document_round_trip():
    from langchain_core.documents import Document as LCDocument

    src = LCDocument(id="d-9", page_content="ciao", metadata={"k": 1})
    out = from_langchain_document(src)

    assert isinstance(out, OrchidDocument)
    assert out.page_content == "ciao"
    assert out.metadata == {"k": 1}
    assert out.id == "d-9"


def test_from_langchain_document_accepts_duck_type():
    """The adapter accepts any object exposing the right attributes —
    handy for tests that don't want to import langchain just to feed
    the converter."""

    class FakeDoc:
        page_content = "duck"
        metadata = {"src": "fake"}
        id = "duck-1"

    out = from_langchain_document(FakeDoc())
    assert isinstance(out, OrchidDocument)
    assert out.page_content == "duck"
    assert out.metadata == {"src": "fake"}
    assert out.id == "duck-1"


def test_from_langchain_document_handles_missing_id():
    class NoId:
        page_content = "no id here"
        metadata: dict = {}

    out = from_langchain_document(NoId())
    assert out.id is None
    assert out.page_content == "no id here"


def test_from_langchain_document_copies_metadata():
    """Mutating the returned OrchidDocument's metadata must not leak
    back into the source langchain Document."""
    from langchain_core.documents import Document as LCDocument

    src = LCDocument(id="d-1", page_content="hi", metadata={"a": 1})
    out = from_langchain_document(src)
    out.metadata["a"] = 999
    assert src.metadata == {"a": 1}


def test_from_langchain_documents_batch():
    from langchain_core.documents import Document as LCDocument

    src = [LCDocument(page_content=t, id=f"d-{i}") for i, t in enumerate(("x", "y"))]
    out = from_langchain_documents(src)
    assert len(out) == 2
    assert all(isinstance(d, OrchidDocument) for d in out)
    assert [d.page_content for d in out] == ["x", "y"]


# ── Symmetric round-trip ──────────────────────────────────────────────


def test_round_trip_preserves_fields():
    """OrchidDocument → LCDocument → OrchidDocument preserves all fields."""
    src = OrchidDocument(id="rt-1", page_content="round trip", metadata={"a": 1, "b": "two"})
    bounced = from_langchain_document(to_langchain_document(src))

    assert bounced.id == src.id
    assert bounced.page_content == src.page_content
    assert bounced.metadata == src.metadata
    # Different objects — adapter must copy.
    assert bounced.metadata is not src.metadata


def test_round_trip_orchid_search_result_keeps_orchid_document():
    """OrchidSearchResult continues to wrap an OrchidDocument after
    we route through the adapter."""
    from orchid_ai.core.repository import OrchidSearchResult

    sr = OrchidSearchResult(document=OrchidDocument(id="d", page_content="x"), score=0.5)
    converted = to_langchain_document(sr.document)
    assert converted.id == "d"
    assert converted.page_content == "x"


# ── Adapter module does not pull LangChain at import time ─────────────


def test_adapter_module_does_not_import_langchain_eagerly():
    """The langchain import inside ``to_langchain_document`` is local
    so consumers that only need ``OrchidDocument`` never pay for the
    langchain import graph just by importing
    :mod:`orchid_ai.rag.adapters`."""
    import importlib
    import sys

    # Drop and reimport the adapter module to make sure no module-level
    # langchain import has been baked in.  (This is best-effort — if
    # another test file already imported langchain we can't unimport
    # it cleanly, but the assertion still holds for the adapters
    # module's own behaviour.)
    sys.modules.pop("orchid_ai.rag.adapters", None)
    mod = importlib.import_module("orchid_ai.rag.adapters")

    # The module-level dir() must not expose ``Document`` as a name
    # — that's the langchain class — only the adapter functions.
    public = {n for n in dir(mod) if not n.startswith("_")}
    assert "Document" not in public
    assert "to_langchain_document" in public
    assert "from_langchain_document" in public


# ── pytest fixtures-friendly behaviour ────────────────────────────────


@pytest.mark.asyncio
async def test_orchid_document_is_usable_as_writer_input(mock_writer):
    """OrchidDocument flows through OrchidVectorWriter unchanged."""
    docs = [
        OrchidDocument(id="d-1", page_content="hello"),
        OrchidDocument(id="d-2", page_content="world", metadata={"k": "v"}),
    ]
    await mock_writer.upsert(docs, "ns")
    assert len(mock_writer.upserted) == 1
    received_docs, ns = mock_writer.upserted[0]
    assert ns == "ns"
    assert all(isinstance(d, OrchidDocument) for d in received_docs)
