"""
Backend factories — name-keyed registries for every pluggable backend (ADR-028).

Replaces the original ``if/elif vector_backend == "qdrant"`` chain with
four registries — one per ABC — so integrators register their custom
backend by calling ``register_*_backend(name, builder)`` from a
composition root before constructing :class:`Orchid`.

Built-ins register on import:

  * vector       — ``null``, ``qdrant``
  * doc_store    — ``null``, ``in_memory``, ``qdrant``
  * graph_store  — ``null``
  * sparse       — ``bm25`` (Stage 1 stub)

Stage 5+ backends (``in_memory`` and ``neo4j`` graph stores, ``splade``
sparse encoder) register from their own modules in their landing stages.

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
from .backends.null import NullDocStore, NullGraphStore, NullVectorReader
from .embeddings import build_embeddings, get_embedding_dimension
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


def build_reader(
    *,
    vector_backend: str = "qdrant",
    **settings: Any,
) -> OrchidVectorReader:
    """Build a vector reader by registry name.

    Raises :class:`ValueError` with the registered names listed when
    ``vector_backend`` is unknown — easier to spot a YAML typo than a
    silent fall-through.
    """
    builder = VECTOR_BACKEND_REGISTRY.get(vector_backend)
    if builder is None:
        raise ValueError(
            f"Unknown vector backend {vector_backend!r}. "
            f"Registered: {sorted(VECTOR_BACKEND_REGISTRY)}. "
            f"Call register_vector_backend({vector_backend!r}, builder) "
            f"before constructing Orchid."
        )
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
        raise ValueError(
            f"Unknown doc store backend {doc_store_backend!r}. "
            f"Registered: {sorted(DOC_STORE_BACKEND_REGISTRY)}. "
            f"Call register_doc_store_backend({doc_store_backend!r}, builder) "
            f"before constructing Orchid."
        )
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
        raise ValueError(
            f"Unknown graph store backend {graph_store_backend!r}. "
            f"Registered: {sorted(GRAPH_STORE_BACKEND_REGISTRY)}. "
            f"Call register_graph_store_backend({graph_store_backend!r}, builder) "
            f"before constructing Orchid."
        )
    return builder(**settings)


# ── Sparse encoder backend ────────────────────────────────────
#
# Sparse encoders use the registry in :mod:`orchid_ai.rag.sparse`
# directly — re-exported here so ADR-028's promise of four matching
# ``register_*_backend`` helpers in one module holds.


def register_sparse_encoder_backend(name: str, cls: type[OrchidSparseEncoder]) -> None:
    """Register a sparse encoder by name (delegates to ``rag.sparse``)."""
    register_sparse_encoder(name, cls)


def build_sparse_encoder(*, sparse_encoder: str = "bm25", **_settings: Any) -> OrchidSparseEncoder:
    """Build a sparse encoder by registry name (delegates to ``rag.sparse``)."""
    return get_sparse_encoder(sparse_encoder)


# ── Built-in backends register on import ──────────────────────


def _build_null_reader(**_settings: Any) -> OrchidVectorReader:
    return NullVectorReader()


def _build_qdrant_reader(
    *,
    qdrant_url: str = "http://qdrant:6333",
    embedding_model: str = "text-embedding-3-small",
    **_settings: Any,
) -> OrchidVectorReader:
    from .backends.qdrant import QdrantRepository

    embeddings = build_embeddings(embedding_model)
    dimension = get_embedding_dimension(embedding_model)
    repo = QdrantRepository(
        url=qdrant_url,
        embeddings=embeddings,
        embedding_dimension=dimension,
    )
    logger.info(
        "[RAG] Using QdrantRepository (url=%s, model=%s, dim=%d)",
        qdrant_url,
        embedding_model,
        dimension,
    )
    return repo


def _build_null_doc_store(**_settings: Any) -> OrchidDocStore:
    return NullDocStore()


def _build_in_memory_doc_store(**_settings: Any) -> OrchidDocStore:
    from .backends.in_memory_doc_store import InMemoryDocStore

    return InMemoryDocStore()


def _build_qdrant_doc_store(
    *,
    qdrant_url: str = "http://qdrant:6333",
    doc_store_collection: str = "__doc_store__",
    **_settings: Any,
) -> OrchidDocStore:
    from .backends.qdrant_doc_store import QdrantDocStore

    return QdrantDocStore(url=qdrant_url, collection_name=doc_store_collection)


def _build_null_graph_store(**_settings: Any) -> OrchidGraphStore:
    return NullGraphStore()


def _build_in_memory_graph_store(**_settings: Any) -> OrchidGraphStore:
    from .backends.in_memory_graph import InMemoryGraphStore

    return InMemoryGraphStore()


def _build_neo4j_graph_store(
    *,
    neo4j_url: str = "bolt://localhost:7687",
    neo4j_user: str = "",
    neo4j_password: str = "",
    neo4j_database: str = "neo4j",
    **_settings: Any,
) -> OrchidGraphStore:
    from .backends.neo4j_graph import Neo4jGraphStore

    return Neo4jGraphStore(
        url=neo4j_url,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )


register_vector_backend("null", _build_null_reader)
register_vector_backend("qdrant", _build_qdrant_reader)
register_doc_store_backend("null", _build_null_doc_store)
register_doc_store_backend("in_memory", _build_in_memory_doc_store)
register_doc_store_backend("qdrant", _build_qdrant_doc_store)
register_graph_store_backend("null", _build_null_graph_store)
register_graph_store_backend("in_memory", _build_in_memory_graph_store)
register_graph_store_backend("neo4j", _build_neo4j_graph_store)


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
