"""
Sparse encoder registry (ADR-025, ADR-028).

Mirrors :mod:`orchid_ai.rag.strategies` and
:mod:`orchid_ai.rag.transformers`: register, get-by-name with safe
fallback, clear for tests.

Stage 4 ships the full BM25-Okapi in-process encoder (lazy IDF tables
per namespace + drift-based avgdl refresh) and the opt-in
:class:`SpladeEncoder` behind the ``splade`` extra
(``pip install orchid-ai[splade]``).  The Splade class lives in
:mod:`orchid_ai.rag.sparse.splade`; constructing it raises
``ImportError`` when the extra isn't installed.
"""

from __future__ import annotations

import logging

from ...core.sparse import OrchidSparseEncoder
from .bm25 import BM25Encoder
from .splade import SpladeEncoder

logger = logging.getLogger(__name__)


_BUILTINS: dict[str, type[OrchidSparseEncoder]] = {
    "bm25": BM25Encoder,
    "splade": SpladeEncoder,
}

SPARSE_ENCODER_REGISTRY: dict[str, type[OrchidSparseEncoder]] = dict(_BUILTINS)


def register_sparse_encoder(name: str, cls: type[OrchidSparseEncoder]) -> None:
    """Register a custom sparse encoder by name."""
    if name in SPARSE_ENCODER_REGISTRY and SPARSE_ENCODER_REGISTRY[name] is not cls:
        logger.warning(
            "[SparseEncoders] '%s' already registered (was %s); overwriting with %s",
            name,
            SPARSE_ENCODER_REGISTRY[name].__name__,
            cls.__name__,
        )
    SPARSE_ENCODER_REGISTRY[name] = cls
    logger.info("[SparseEncoders] Registered '%s' → %s", name, cls.__name__)


def clear_sparse_encoders() -> None:
    """Reset to built-in encoders (useful for test isolation)."""
    SPARSE_ENCODER_REGISTRY.clear()
    SPARSE_ENCODER_REGISTRY.update(_BUILTINS)


def get_sparse_encoder(name: str) -> OrchidSparseEncoder:
    """Look up and instantiate a sparse encoder by name.

    Falls back to ``"bm25"`` when the name is unknown — every Hybrid
    setup is expected to use either the built-in BM25 or a custom
    encoder, so a typo degrades to a no-op stub instead of crashing.
    """
    cls = SPARSE_ENCODER_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown sparse encoder '%s', falling back to 'bm25'", name)
        cls = SPARSE_ENCODER_REGISTRY.get("bm25", BM25Encoder)
    return cls()


__all__ = [
    "BM25Encoder",
    "OrchidSparseEncoder",
    "SPARSE_ENCODER_REGISTRY",
    "SpladeEncoder",
    "clear_sparse_encoders",
    "get_sparse_encoder",
    "register_sparse_encoder",
]
