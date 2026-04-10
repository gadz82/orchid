"""
RAG factory — builds the concrete VectorStoreRepository at runtime.

Chosen via ``Settings.vector_backend``:
  - ``qdrant`` → QdrantRepository (PoC)
  - ``aoss``   → AOSSRepository (production, not yet implemented)
  - ``null``   → NullVectorReader (no vector DB)

This is the only place where concrete vector DB clients are imported.
"""

from __future__ import annotations

import logging

from ..core.repository import VectorReader
from .embeddings import LiteLLMEmbedder
from .null import NullVectorReader

logger = logging.getLogger(__name__)


def build_reader(
    *,
    vector_backend: str = "qdrant",
    qdrant_url: str = "http://qdrant:6333",
    embedding_model: str = "text-embedding-3-small",
) -> VectorReader:
    """
    Factory that returns the right VectorReader based on config.

    Returns a full VectorStoreRepository (which implements VectorReader)
    for backends that support writes, or NullVectorReader as a fallback.
    """
    if vector_backend == "null":
        logger.info("[RAG] Using NullVectorReader (no vector DB)")
        return NullVectorReader()

    if vector_backend == "qdrant":
        from .backends.qdrant import QdrantRepository

        embedder = LiteLLMEmbedder(embedding_model)
        repo = QdrantRepository(
            url=qdrant_url,
            embedder=embedder,
        )
        logger.info(
            "[RAG] Using QdrantRepository (url=%s, model=%s, dim=%d)",
            qdrant_url,
            embedding_model,
            embedder.dimension,
        )
        return repo

    if vector_backend == "aoss":
        raise NotImplementedError("AOSSRepository is planned for production — not yet implemented")

    raise ValueError(f"Unknown vector backend: {vector_backend!r}")
