"""
``HybridRetrieval`` — dense + sparse hybrid search (ADR-025).

The strategy issues two parallel retrievals — one against the dense
embedding lane (``reader.retrieve``) and one against the sparse /
lexical lane (``reader.retrieve_sparse``) — then fuses the rankings
via Reciprocal Rank Fusion (RRF, default) or weighted-linear fusion.

When the underlying backend doesn't support sparse retrieval the
strategy detects ``NotImplementedError`` from the sparse lane and
gracefully degrades to dense-only, logging a one-line warning.

Sparse query encoding uses a registry-resolved
:class:`OrchidSparseEncoder` (default ``bm25``).  Note that the
strategy's encoder instance is independent of any encoder used at
ingestion time — Stage 4's BM25 IDF state lives per-instance.  In a
production deployment integrators inject a single shared encoder via
``register_retrieval_strategy("hybrid", ...)`` so the writer and
reader sides agree on the IDF table.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from ...core.scopes import OrchidRAGScope
from ...core.sparse import OrchidSparseEncoder
from ..sparse import BM25Encoder, get_sparse_encoder

logger = logging.getLogger(__name__)


_FusionAlgo = Literal["rrf", "linear"]


class HybridRetrieval(OrchidRetrievalStrategy):
    """Dense + sparse retrieval with RRF or linear fusion."""

    def __init__(
        self,
        *,
        sparse_encoder: OrchidSparseEncoder | None = None,
        sparse_weight: float = 0.4,
        fusion: _FusionAlgo = "rrf",
        rrf_k: int = 60,
        lane_multiplier: int = 3,
    ) -> None:
        if not 0.0 <= sparse_weight <= 1.0:
            raise ValueError(f"sparse_weight must be in [0, 1]; got {sparse_weight}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0; got {rrf_k}")
        if lane_multiplier < 1:
            raise ValueError(f"lane_multiplier must be >= 1; got {lane_multiplier}")
        self._sparse_encoder = sparse_encoder or BM25Encoder()
        self._sparse_weight = sparse_weight
        self._fusion = fusion
        self._rrf_k = rrf_k
        # Each lane fetches ``k * lane_multiplier`` candidates so the
        # fusion has enough headroom to surface dense-only or
        # sparse-only matches into the top-k.
        self._lane_multiplier = lane_multiplier

    @classmethod
    def from_config(cls, config: Any) -> "HybridRetrieval":
        """Read ``config.hybrid`` (sparse_encoder name, fusion, weights)."""
        hybrid_cfg = getattr(config, "hybrid", None) if config is not None else None
        if hybrid_cfg is None:
            return cls()
        encoder_name = getattr(hybrid_cfg, "sparse_encoder", "bm25")
        try:
            encoder = get_sparse_encoder(encoder_name)
        except Exception as exc:  # pragma: no cover — registry resolution edge case
            logger.warning(
                "[HybridRetrieval] Sparse encoder %r unavailable (%s); falling back to bm25",
                encoder_name,
                exc,
            )
            encoder = BM25Encoder()
        return cls(
            sparse_encoder=encoder,
            sparse_weight=getattr(hybrid_cfg, "sparse_weight", 0.4),
            fusion=getattr(hybrid_cfg, "fusion", "rrf"),
            rrf_k=getattr(hybrid_cfg, "rrf_k", 60),
        )

    async def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        scope: OrchidRAGScope,
        k: int,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        doc_store: OrchidDocStore | None = None,
        transformers: list[OrchidQueryTransformer] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        # Encode sparse first — if encoding itself fails (rare) we
        # still want the dense lane to run.
        sparse_query = None
        try:
            sparse_query = await self._sparse_encoder.encode_query(query, namespace=namespace)
        except Exception as exc:
            logger.warning("[HybridRetrieval] Sparse encoder failed: %s — dense-only", exc)

        lane_k = k * self._lane_multiplier
        dense_task = reader.retrieve(
            query=query,
            namespace=namespace,
            k=lane_k,
            scope=scope,
            metadata_filters=metadata_filters,
        )
        if sparse_query is not None:
            sparse_task = reader.retrieve_sparse(
                query_sparse=sparse_query,
                namespace=namespace,
                k=lane_k,
                scope=scope,
                metadata_filters=metadata_filters,
            )
            results = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)
        else:
            results = [await asyncio.gather(dense_task, return_exceptions=True), [None]]
            # Flatten the gather-of-one back to ``[dense_result, marker]``.
            results = [results[0][0], None]

        dense, sparse = results[0], results[1]

        if isinstance(dense, BaseException):
            logger.warning("[HybridRetrieval] Dense retrieval failed: %s", dense)
            dense = []
        if isinstance(sparse, NotImplementedError):
            logger.warning("[HybridRetrieval] Backend lacks sparse support — dense-only fallback")
            sparse = []
        elif isinstance(sparse, BaseException):
            logger.warning("[HybridRetrieval] Sparse retrieval failed: %s", sparse)
            sparse = []
        elif sparse is None:
            # Sparse lane skipped (encoder failed earlier).
            sparse = []

        if not sparse:
            # Dense-only short-circuit — preserve original ordering.
            return list(dense)[:k]

        if self._fusion == "rrf":
            return _rrf_merge(dense, sparse, k=k, rrf_k=self._rrf_k)
        return _linear_merge(dense, sparse, k=k, sparse_weight=self._sparse_weight)


def _doc_id(sr: OrchidSearchResult) -> str:
    """Stable identifier for fusion dedupe — falls back to a content prefix
    when the document carries no explicit id."""
    return sr.document.id or sr.document.page_content[:100]


def _rrf_merge(
    dense: list[OrchidSearchResult],
    sparse: list[OrchidSearchResult],
    *,
    k: int,
    rrf_k: int,
) -> list[OrchidSearchResult]:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""
    scores: dict[str, float] = {}
    docs: dict[str, OrchidSearchResult] = {}
    for ranking in (dense, sparse):
        for rank, sr in enumerate(ranking, start=1):
            doc_id = _doc_id(sr)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(doc_id, sr)
    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    out: list[OrchidSearchResult] = []
    for doc_id in sorted_ids[:k]:
        sr = docs[doc_id]
        out.append(OrchidSearchResult(document=sr.document, score=scores[doc_id]))
    return out


def _linear_merge(
    dense: list[OrchidSearchResult],
    sparse: list[OrchidSearchResult],
    *,
    k: int,
    sparse_weight: float,
) -> list[OrchidSearchResult]:
    """Weighted linear fusion of normalised lane scores.

    Each lane is min-max normalised to ``[0, 1]`` so the weights are
    interpretable across backends with arbitrary score ranges.
    """
    dense_norm = _minmax_normalise(dense)
    sparse_norm = _minmax_normalise(sparse)
    dense_weight = 1.0 - sparse_weight

    scores: dict[str, float] = {}
    docs: dict[str, OrchidSearchResult] = {}
    for sr, score in dense_norm:
        doc_id = _doc_id(sr)
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * score
        docs.setdefault(doc_id, sr)
    for sr, score in sparse_norm:
        doc_id = _doc_id(sr)
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight * score
        docs.setdefault(doc_id, sr)
    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [OrchidSearchResult(document=docs[doc_id].document, score=scores[doc_id]) for doc_id in sorted_ids[:k]]


def _minmax_normalise(
    results: list[OrchidSearchResult],
) -> list[tuple[OrchidSearchResult, float]]:
    """Min-max normalise scores into ``[0, 1]`` so lane weights compose."""
    if not results:
        return []
    scores = [sr.score for sr in results]
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng <= 0:
        return [(sr, 1.0) for sr in results]
    return [(sr, (sr.score - lo) / rng) for sr in results]
