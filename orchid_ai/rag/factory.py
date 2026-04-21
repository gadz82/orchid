"""
RAG factory — builds the concrete OrchidVectorStoreRepository at runtime.

Chosen via ``Settings.vector_backend``:
  - ``qdrant`` → QdrantRepository (PoC)
  - ``aoss``   → AOSSRepository (production, not yet implemented)
  - ``null``   → NullVectorReader (no vector DB)

This is the only place where concrete vector DB clients are imported.
"""

from __future__ import annotations

import logging

from ..core.repository import OrchidVectorReader
from .embeddings import build_embeddings, get_embedding_dimension
from .null import NullVectorReader

logger = logging.getLogger(__name__)


def build_reader(
    *,
    vector_backend: str = "qdrant",
    qdrant_url: str = "http://qdrant:6333",
    embedding_model: str = "text-embedding-3-small",
) -> OrchidVectorReader:
    """
    Factory that returns the right OrchidVectorReader based on config.

    Returns a full OrchidVectorStoreRepository (which implements OrchidVectorReader)
    for backends that support writes, or NullVectorReader as a fallback.
    """
    if vector_backend == "null":
        logger.info("[RAG] Using NullVectorReader (no vector DB)")
        return NullVectorReader()

    if vector_backend == "qdrant":
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

    if vector_backend == "aoss":
        raise NotImplementedError("AOSSRepository is planned for production — not yet implemented")

    raise ValueError(f"Unknown vector backend: {vector_backend!r}")
