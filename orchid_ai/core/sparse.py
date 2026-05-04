"""
Sparse vector primitives — Open/Closed extension point for hybrid search (ADR-025).

Hybrid retrieval combines dense embeddings with a sparse / lexical lane (BM25,
Splade, ...). The sparse lane needs:

  * a vector representation that vector backends can index alongside the dense one
    (:class:`OrchidSparseVector`)
  * a way to encode arbitrary text into that representation
    (:class:`OrchidSparseEncoder`)

Both live in ``core/`` because every other module — strategies, backends,
runtime injection — depends on them.  The ABC stays free of any concrete
encoder dependency (BM25, Splade, ...) and uses ``Any`` for return types
that may vary per encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple


class OrchidSparseVector(NamedTuple):
    """A sparse vector — non-zero indices and their float values.

    The pairing of ``indices`` and ``values`` follows the standard COO
    layout: ``indices[i]`` carries weight ``values[i]``.  Both lists must
    be the same length.  Empty lists represent the zero vector and are a
    valid encoding (e.g. for a query whose tokens are all out-of-vocab).
    """

    indices: list[int]
    values: list[float]


class OrchidSparseEncoder(ABC):
    """Encode text into :class:`OrchidSparseVector` values.

    Two methods on purpose: documents and queries are encoded with
    different conventions in classical IR (e.g. BM25 weights documents
    by TF×IDF, queries by IDF only).  Implementations are free to make
    the two equivalent.

    The optional ``namespace`` keyword threads through both methods so
    encoders can maintain per-namespace state (e.g. BM25's lazy IDF
    table).  Implementations that don't need it can ignore the param;
    ``None`` is treated as a single shared "default" namespace.

    Implementations must be **stateless across encoders** — any per-corpus
    state (IDF tables, vocabulary, ...) is owned by the encoder instance
    and re-initialised when the encoder is replaced.
    """

    @abstractmethod
    async def encode_documents(
        self,
        texts: list[str],
        namespace: str | None = None,
    ) -> list[OrchidSparseVector]:
        """Encode a batch of document texts.

        Returns one :class:`OrchidSparseVector` per input in the same order.
        """
        ...

    @abstractmethod
    async def encode_query(
        self,
        text: str,
        namespace: str | None = None,
    ) -> OrchidSparseVector:
        """Encode a single query."""
        ...
