"""Tests for ``HybridRetrieval``."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from orchid_ai.config.schema import OrchidRetrievalConfig
from orchid_ai.config.schema_rag import OrchidHybridConfig
from orchid_ai.core.repository import OrchidSearchResult
from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.core.sparse import OrchidSparseEncoder, OrchidSparseVector
from orchid_ai.rag.strategies.hybrid import (
    HybridRetrieval,
    _linear_merge,
    _rrf_merge,
)


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


def _result(doc_id: str, score: float, content: str | None = None) -> OrchidSearchResult:
    return OrchidSearchResult(
        document=Document(id=doc_id, page_content=content or doc_id, metadata={}),
        score=score,
    )


class _StaticEncoder(OrchidSparseEncoder):
    """Test double — returns a fixed sparse vector regardless of input."""

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self._indices = indices
        self._values = values

    async def encode_documents(self, texts, namespace=None):
        return [OrchidSparseVector(indices=self._indices, values=self._values) for _ in texts]

    async def encode_query(self, text, namespace=None):
        return OrchidSparseVector(indices=self._indices, values=self._values)


class _ZeroEncoder(OrchidSparseEncoder):
    async def encode_documents(self, texts, namespace=None):
        return [OrchidSparseVector(indices=[], values=[]) for _ in texts]

    async def encode_query(self, text, namespace=None):
        return OrchidSparseVector(indices=[], values=[])


class TestExactTokenMatch:
    @pytest.mark.asyncio
    async def test_sparse_lane_promotes_exact_token_match(self):
        """The classic hybrid win: an exact-token match (e.g. SKU-A1234) is
        present only in the sparse lane's top-k; the dense lane misses it.
        After fusion, the exact-match doc must appear in the top-k."""
        reader = MagicMock()
        # Dense lane returns semantically-related docs but not the SKU.
        # Sparse lane finds the SKU document.
        reader.retrieve = AsyncMock(
            return_value=[
                _result("similar-1", 0.85),
                _result("similar-2", 0.80),
            ]
        )
        reader.retrieve_sparse = AsyncMock(return_value=[_result("sku-a1234", 5.0)])

        encoder = _StaticEncoder(indices=[1, 2], values=[1.0, 1.0])
        results = await HybridRetrieval(sparse_encoder=encoder).retrieve(
            query="SKU-A1234",
            namespace="kb",
            scope=_scope(),
            k=3,
            reader=reader,
        )
        ids = [r.document.id for r in results]
        assert "sku-a1234" in ids


class TestNotImplementedFallback:
    @pytest.mark.asyncio
    async def test_dense_only_when_sparse_unsupported(self, caplog):
        """Backend without sparse support → dense-only with warning."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.9), _result("b", 0.8)])
        reader.retrieve_sparse = AsyncMock(side_effect=NotImplementedError("legacy"))

        encoder = _StaticEncoder(indices=[1], values=[1.0])
        with caplog.at_level("WARNING"):
            results = await HybridRetrieval(sparse_encoder=encoder).retrieve(
                query="q", namespace="kb", scope=_scope(), k=2, reader=reader
            )
        assert [r.document.id for r in results] == ["a", "b"]
        assert any("sparse" in rec.message.lower() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_dense_only_when_sparse_returns_empty(self):
        """Empty sparse hits → dense-only short-circuit (no fusion needed)."""
        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.9)])
        reader.retrieve_sparse = AsyncMock(return_value=[])

        encoder = _StaticEncoder(indices=[1], values=[1.0])
        results = await HybridRetrieval(sparse_encoder=encoder).retrieve(
            query="q", namespace="kb", scope=_scope(), k=5, reader=reader
        )
        assert [r.document.id for r in results] == ["a"]


class TestRRFFusion:
    def test_rrf_tracks_rank_not_score(self):
        """RRF gives equal weight to each lane's rank — score magnitudes
        don't dominate."""
        # Dense lane has score 0.9 vs sparse lane's score 5.0; RRF should
        # still surface the top sparse hit even though its raw score
        # would dominate a linear sum.
        dense = [_result("dense-top", 0.9), _result("shared", 0.85)]
        sparse = [_result("sparse-top", 5.0), _result("shared", 4.5)]
        merged = _rrf_merge(dense, sparse, k=3, rrf_k=60)
        ids = [r.document.id for r in merged]
        # "shared" appears in both rankings → highest RRF score.
        assert ids[0] == "shared"

    def test_rrf_preserves_unique_lane_hits(self):
        dense = [_result("only-dense", 0.5)]
        sparse = [_result("only-sparse", 5.0)]
        merged = _rrf_merge(dense, sparse, k=2, rrf_k=60)
        ids = {r.document.id for r in merged}
        assert ids == {"only-dense", "only-sparse"}


class TestLinearFusion:
    def test_linear_respects_sparse_weight(self):
        """sparse_weight=1.0 → sparse lane wins; sparse_weight=0.0 → dense wins."""
        dense = [_result("dense", 0.9), _result("shared", 0.5)]
        sparse = [_result("sparse", 5.0), _result("shared", 4.5)]

        sparse_only = _linear_merge(dense, sparse, k=2, sparse_weight=1.0)
        dense_only = _linear_merge(dense, sparse, k=2, sparse_weight=0.0)
        assert sparse_only[0].document.id == "sparse"
        assert dense_only[0].document.id == "dense"

    def test_balanced_weight_blends(self):
        """Mid-weight should consider both lanes."""
        dense = [_result("a", 0.9), _result("b", 0.4)]
        sparse = [_result("a", 0.5), _result("b", 5.0)]
        # With sparse_weight=0.5, each doc gets contributions from both
        # lanes after normalisation.  "a" tops dense; "b" tops sparse.
        merged = _linear_merge(dense, sparse, k=2, sparse_weight=0.5)
        ids = {r.document.id for r in merged}
        assert ids == {"a", "b"}


class TestFromConfig:
    def test_reads_hybrid_block(self):
        cfg = OrchidRetrievalConfig(
            hybrid=OrchidHybridConfig(
                sparse_encoder="bm25",
                fusion="linear",
                sparse_weight=0.7,
                rrf_k=42,
            )
        )
        strategy = HybridRetrieval.from_config(cfg)
        assert strategy._fusion == "linear"
        assert strategy._sparse_weight == 0.7
        assert strategy._rrf_k == 42

    def test_default_when_no_config(self):
        strategy = HybridRetrieval.from_config(None)
        assert strategy._fusion == "rrf"
        assert strategy._sparse_weight == 0.4
        assert strategy._rrf_k == 60


class TestPreStrategyTransformersSkipped:
    @pytest.mark.asyncio
    async def test_pre_strategy_transformers_ignored(self):
        """Hybrid doesn't run pre_strategy transformers — those belong to
        the agent-entry path."""
        from orchid_ai.core.retrieval import OrchidQueryTransformer

        class _PreStrategyDouble(OrchidQueryTransformer):
            pre_strategy: ClassVar[bool] = True

            async def transform(self, query, *, chat_model, history=None):  # pragma: no cover
                raise AssertionError("HybridRetrieval must not run pre_strategy transformers")

        reader = MagicMock()
        reader.retrieve = AsyncMock(return_value=[_result("a", 0.5)])
        reader.retrieve_sparse = AsyncMock(side_effect=NotImplementedError("legacy"))

        encoder = _StaticEncoder(indices=[1], values=[1.0])
        await HybridRetrieval(sparse_encoder=encoder).retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            k=2,
            reader=reader,
            transformers=[_PreStrategyDouble()],
        )


class TestValidation:
    def test_invalid_sparse_weight_raises(self):
        with pytest.raises(ValueError, match="sparse_weight"):
            HybridRetrieval(sparse_weight=1.5)

    def test_zero_rrf_k_raises(self):
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetrieval(rrf_k=0)

    def test_zero_lane_multiplier_raises(self):
        with pytest.raises(ValueError, match="lane_multiplier"):
            HybridRetrieval(lane_multiplier=0)
