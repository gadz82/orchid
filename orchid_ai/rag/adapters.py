"""
LangChain ``Document`` ↔ :class:`OrchidDocument` adapter.

The framework's canonical document model is :class:`OrchidDocument`, a
stdlib dataclass that lives in :mod:`orchid_ai.core.repository` (the
zero-dependency layer — see ``core/AGENTS.md``).  RAG backends and
LangChain-compatible glue code (the
:class:`~orchid_ai.rag.retriever.OrchidRetriever` ``BaseRetriever``
wrapper, custom integrator backends that delegate to LangChain
vector-stores, …) need a thin conversion layer to LangChain's
``langchain_core.documents.Document``.

This module is the single allowed crossing point.  It lives under
:mod:`orchid_ai.rag` so the architectural lint
(``tests/test_dependency_boundaries.py``) can permit
``langchain_core`` here without weakening the rule for ``core/``.

All adapter functions are pure — they construct fresh objects without
mutating either side.  Metadata dicts are copied (shallow ``dict()``)
so callers who later mutate one side don't surprise the other.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..core.repository import OrchidDocument

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    from langchain_core.documents import Document as _LCDocument


__all__ = [
    "from_langchain_document",
    "from_langchain_documents",
    "to_langchain_document",
    "to_langchain_documents",
]


def to_langchain_document(doc: OrchidDocument) -> _LCDocument:
    """Convert an :class:`OrchidDocument` to a ``langchain_core.Document``.

    The langchain Document is constructed lazily — the import happens
    inside the function body so importing
    :mod:`orchid_ai.rag.adapters` doesn't pull LangChain into module
    load time for callers that only need :class:`OrchidDocument`.
    """
    from langchain_core.documents import Document  # local import — see docstring

    return Document(
        id=doc.id,
        page_content=doc.page_content,
        metadata=dict(doc.metadata or {}),
    )


def from_langchain_document(doc: Any) -> OrchidDocument:
    """Convert a ``langchain_core.Document`` (or any duck-compatible
    object) to an :class:`OrchidDocument`.

    Accepts ``Any`` so callers can pass anything exposing
    ``page_content``, ``metadata``, and ``id`` (or just ``page_content``
    with sensible defaults).  This keeps the adapter usable from tests
    and integrators that hand in lightweight stand-ins without the
    real LangChain dependency.
    """
    return OrchidDocument(
        id=getattr(doc, "id", None),
        page_content=getattr(doc, "page_content", ""),
        metadata=dict(getattr(doc, "metadata", None) or {}),
    )


def to_langchain_documents(docs: Iterable[OrchidDocument]) -> list[_LCDocument]:
    """Batch wrapper around :func:`to_langchain_document`."""
    return [to_langchain_document(d) for d in docs]


def from_langchain_documents(docs: Iterable[Any]) -> list[OrchidDocument]:
    """Batch wrapper around :func:`from_langchain_document`."""
    return [from_langchain_document(d) for d in docs]
