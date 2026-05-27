"""
Backend factories — name-keyed registries for every pluggable backend.

Replaces the original ``if/elif vector_backend == "qdrant"`` chain with
four registries — one per ABC — so integrators register their custom
backend by calling ``register_*_backend(name, builder)`` from a
composition root before constructing :class:`Orchid`.

Built-ins register on import:

  * vector       — ``null``
  * doc_store    — ``null``, ``in_memory``
  * graph_store  — ``null``, ``in_memory``
  * sparse       — ``bm25``

External backends (Qdrant, Neo4j, ChromaDB) register automatically via
Python entry points when their plugin package is installed.

The factories are deliberately permissive about ``**settings`` — each
builder picks only the kwargs it needs.  This keeps ``build_reader``'s
signature stable: existing callers (``bootstrap._prepare_reader``) keep
passing ``vector_backend=…, qdrant_url=…, embedding_model=…`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..core.doc_store import OrchidDocStore
from ..core.graph_store import OrchidGraphStore
from ..core.repository import OrchidVectorReader
from ..core.sparse import OrchidSparseEncoder
from ..plugins import iter_entry_point_plugins
from .backends.null import NullDocStore, NullGraphStore, NullVectorReader
from .sparse import get_sparse_encoder, register_sparse_encoder  # re-exported below

logger = logging.getLogger(__name__)


# Builders accept arbitrary keyword settings and return one of the four
# backend ABCs.  Loose typing lets each builder pick the kwargs it needs
# without forcing every caller through a heavyweight settings object.
VectorBackendBuilder = Callable[..., OrchidVectorReader]
DocStoreBackendBuilder = Callable[..., OrchidDocStore]
GraphStoreBackendBuilder = Callable[..., OrchidGraphStore]
SparseEncoderBackendBuilder = Callable[..., OrchidSparseEncoder]


VECTOR_BACKEND_REGISTRY: dict[str, VectorBackendBuilder] = {}
DOC_STORE_BACKEND_REGISTRY: dict[str, DocStoreBackendBuilder] = {}
GRAPH_STORE_BACKEND_REGISTRY: dict[str, GraphStoreBackendBuilder] = {}


# ── Vector backend ────────────────────────────────────────────


def register_vector_backend(name: str, builder: VectorBackendBuilder) -> None:
    """Register a vector backend by name.

    Integrators call this in their composition root before constructing
    :class:`Orchid`.  Logs a warning when an integrator overwrites a
    built-in name.
    """
    if name in VECTOR_BACKEND_REGISTRY and VECTOR_BACKEND_REGISTRY[name] is not builder:
        logger.warning("[VectorBackends] '%s' already registered; overwriting", name)
    VECTOR_BACKEND_REGISTRY[name] = builder
    logger.debug("[VectorBackends] Registered '%s'", name)


_BACKEND_PACKAGE_HINTS = {
    "qdrant": "orchid-rag-qdrant",
    "neo4j": "orchid-rag-neo4j",
    "chroma": "orchid-rag-chroma",
}


def _format_missing_backend_error(name: str, backend_type: str) -> str:
    hint = _BACKEND_PACKAGE_HINTS.get(name)
    lines = [
        f"Unknown {backend_type} backend {name!r}.",
    ]
    if hint:
        lines.append(f"Install the missing plugin: pip install {hint}")
    lines.append(
        f"Registered built-ins: {sorted(VECTOR_BACKEND_REGISTRY)}. "
        f"Call register_{backend_type}_backend({name!r}, builder) "
        f"before constructing Orchid."
    )
    return " ".join(lines)


def build_reader(
    *,
    vector_backend: str = "null",
    **settings: Any,
) -> OrchidVectorReader:
    """Build a vector reader by registry name.

    Raises :class:`ValueError` with the registered names listed when
    ``vector_backend`` is unknown — easier to spot a YAML typo than a
    silent fall-through.

    **Default is ``"null"``** (no vector database).  This is intentional:
    when no vector backend is configured, retrieval returns an empty result
    set and logs a warning at retrieval time.  The application continues
    to function — agents respond without RAG context — so a missing plugin
    or misconfigured backend does not crash the process.  Operators will
    see ``[NullVectorReader] retrieve() called — no vector backend configured``
    in the logs, making the degradation visible without breaking requests.
    """
    builder = VECTOR_BACKEND_REGISTRY.get(vector_backend)
    if builder is None:
        raise ValueError(_format_missing_backend_error(vector_backend, "vector"))
    return builder(**settings)


# ── Doc store backend ─────────────────────────────────────────


def register_doc_store_backend(name: str, builder: DocStoreBackendBuilder) -> None:
    """Register a doc store backend by name."""
    if name in DOC_STORE_BACKEND_REGISTRY and DOC_STORE_BACKEND_REGISTRY[name] is not builder:
        logger.warning("[DocStoreBackends] '%s' already registered; overwriting", name)
    DOC_STORE_BACKEND_REGISTRY[name] = builder
    logger.debug("[DocStoreBackends] Registered '%s'", name)


def build_doc_store(*, doc_store_backend: str = "null", **settings: Any) -> OrchidDocStore:
    """Build a doc store by registry name."""
    builder = DOC_STORE_BACKEND_REGISTRY.get(doc_store_backend)
    if builder is None:
        raise ValueError(_format_missing_backend_error(doc_store_backend, "doc_store"))
    return builder(**settings)


# ── Graph store backend ───────────────────────────────────────


def register_graph_store_backend(name: str, builder: GraphStoreBackendBuilder) -> None:
    """Register a graph store backend by name."""
    if name in GRAPH_STORE_BACKEND_REGISTRY and GRAPH_STORE_BACKEND_REGISTRY[name] is not builder:
        logger.warning("[GraphStoreBackends] '%s' already registered; overwriting", name)
    GRAPH_STORE_BACKEND_REGISTRY[name] = builder
    logger.debug("[GraphStoreBackends] Registered '%s'", name)


def build_graph_store(*, graph_store_backend: str = "null", **settings: Any) -> OrchidGraphStore:
    """Build a graph store by registry name."""
    builder = GRAPH_STORE_BACKEND_REGISTRY.get(graph_store_backend)
    if builder is None:
        raise ValueError(_format_missing_backend_error(graph_store_backend, "graph_store"))
    return builder(**settings)


# ── Sparse encoder backend ────────────────────────────────────
#
# Sparse encoders use the registry in :mod:`orchid_ai.rag.sparse`
# directly — re-exported here so the four matching
# ``register_*_backend`` helpers all live in one module.


def register_sparse_encoder_backend(name: str, cls: type[OrchidSparseEncoder]) -> None:
    """Register a sparse encoder by name (delegates to ``rag.sparse``)."""
    register_sparse_encoder(name, cls)


def build_sparse_encoder(*, sparse_encoder: str = "bm25", **_settings: Any) -> OrchidSparseEncoder:
    """Build a sparse encoder by registry name (delegates to ``rag.sparse``)."""
    return get_sparse_encoder(sparse_encoder)


# ── Built-in backends register on import ──────────────────────


def _build_null_reader(**_settings: Any) -> OrchidVectorReader:
    return NullVectorReader()


def _build_null_doc_store(**_settings: Any) -> OrchidDocStore:
    return NullDocStore()


def _build_in_memory_doc_store(**_settings: Any) -> OrchidDocStore:
    from .backends.in_memory_doc_store import InMemoryDocStore

    return InMemoryDocStore()


def _build_null_graph_store(**_settings: Any) -> OrchidGraphStore:
    return NullGraphStore()


def _build_in_memory_graph_store(**_settings: Any) -> OrchidGraphStore:
    from .backends.in_memory_graph import InMemoryGraphStore

    return InMemoryGraphStore()


register_vector_backend("null", _build_null_reader)
register_doc_store_backend("null", _build_null_doc_store)
register_doc_store_backend("in_memory", _build_in_memory_doc_store)
register_graph_store_backend("null", _build_null_graph_store)
register_graph_store_backend("in_memory", _build_in_memory_graph_store)


# ── Load external backends via entry points ───────────────────
# Called from :func:`orchid_ai.plugins.lazy_init_plugins` on
# first :class:`orchid_ai.Orchid` construction.


def _load_entry_point_backends() -> None:
    for name, register_fn in iter_entry_point_plugins("orchid.vector_backends"):
        try:
            register_fn()
        except Exception as exc:
            logger.warning("[VectorBackends] Failed to load plugin '%s': %s", name, exc)
    for name, register_fn in iter_entry_point_plugins("orchid.doc_store_backends"):
        try:
            register_fn()
        except Exception as exc:
            logger.warning("[DocStoreBackends] Failed to load plugin '%s': %s", name, exc)
    for name, register_fn in iter_entry_point_plugins("orchid.graph_store_backends"):
        try:
            register_fn()
        except Exception as exc:
            logger.warning("[GraphStoreBackends] Failed to load plugin '%s': %s", name, exc)


__all__ = [
    "DOC_STORE_BACKEND_REGISTRY",
    "DocStoreBackendBuilder",
    "GRAPH_STORE_BACKEND_REGISTRY",
    "GraphStoreBackendBuilder",
    "SparseEncoderBackendBuilder",
    "VECTOR_BACKEND_REGISTRY",
    "VectorBackendBuilder",
    "build_doc_store",
    "build_graph_store",
    "build_reader",
    "build_sparse_encoder",
    "register_doc_store_backend",
    "register_graph_store_backend",
    "register_sparse_encoder_backend",
    "register_vector_backend",
]
