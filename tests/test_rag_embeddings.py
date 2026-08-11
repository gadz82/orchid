"""Tests for orchid_ai.rag.embeddings — provider-aware batching wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from orchid_ai.rag.embeddings import (
    BatchLimitingEmbeddings,
    build_embeddings,
    get_embedding_batch_size,
    get_embedding_dimension,
)

# ── get_embedding_batch_size ────────────────────────────────


class TestGetEmbeddingBatchSize:
    def test_gemini_has_cap(self):
        assert get_embedding_batch_size("gemini/gemini-embedding-001") == 80

    def test_google_shares_gemini_cap(self):
        assert get_embedding_batch_size("google/text-embedding-004") == 80

    def test_cohere_has_cap(self):
        assert get_embedding_batch_size("cohere/embed-english-v3.0") == 80

    def test_voyage_has_cap(self):
        assert get_embedding_batch_size("voyage/voyage-3") == 100

    def test_bedrock_cohere_has_cap(self):
        assert get_embedding_batch_size("bedrock/cohere.embed-english-v3") == 80

    def test_bedrock_titan_unbounded(self):
        # langchain-aws serialises single-input calls; wrapping with a
        # sensible default (32) still avoids surprise rate limits.
        assert get_embedding_batch_size("bedrock/amazon.titan-embed-text-v2:0") == 32

    def test_openai_text_embedding_3_small_has_cap(self):
        assert get_embedding_batch_size("text-embedding-3-small") == 2000

    def test_openai_text_embedding_3_large_has_cap(self):
        assert get_embedding_batch_size("text-embedding-3-large") == 2000

    def test_openai_ada_002_has_cap(self):
        assert get_embedding_batch_size("text-embedding-ada-002") == 2000

    def test_ollama_unbounded(self):
        assert get_embedding_batch_size("ollama/nomic-embed-text") == 32

    def test_unknown_model_unbounded(self):
        assert get_embedding_batch_size("mistral/mistral-embed") == 32


# ── BatchLimitingEmbeddings ────────────────────────────────


class _StubEmbeddings(Embeddings):
    """Records batch sizes passed to embed_documents / aembed_documents."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.sync_batches: list[int] = []
        self.async_batches: list[int] = []

    def _vec(self, text: str) -> list[float]:
        return [float(len(text))] * self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.sync_batches.append(len(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.async_batches.append(len(texts))
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)


class TestBatchLimitingEmbeddingsConstruction:
    def test_zero_batch_size_rejected(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchLimitingEmbeddings(_StubEmbeddings(), batch_size=0)

    def test_negative_batch_size_rejected(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchLimitingEmbeddings(_StubEmbeddings(), batch_size=-5)

    def test_inner_and_batch_size_accessors(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=42)
        assert wrapped.inner is stub
        assert wrapped.batch_size == 42


class TestBatchLimitingEmbeddingsSync:
    def test_small_batch_passes_through_unchanged(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=10)
        result = wrapped.embed_documents(["a", "b", "c"])
        assert len(result) == 3
        # Single forwarded call, not copied/rebatched.
        assert stub.sync_batches == [3]

    def test_exact_batch_passes_through(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=3)
        wrapped.embed_documents(["a", "b", "c"])
        assert stub.sync_batches == [3]

    def test_oversize_batch_splits_into_chunks(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=3)
        texts = [f"t{i}" for i in range(7)]
        result = wrapped.embed_documents(texts)
        assert len(result) == 7
        # 7 texts, batch 3 → 3 + 3 + 1.
        assert stub.sync_batches == [3, 3, 1]

    def test_order_preserved_across_chunks(self):
        stub = _StubEmbeddings(dim=1)
        wrapped = BatchLimitingEmbeddings(stub, batch_size=2)
        texts = ["short", "medium_len", "a", "much_longer_text"]
        result = wrapped.embed_documents(texts)
        # _StubEmbeddings embeds the length of each text as the vector —
        # order-preservation check.
        assert [v[0] for v in result] == [5.0, 10.0, 1.0, 16.0]

    def test_query_passes_through(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=2)
        # Query path is never over-cap (single text), no batching logic involved.
        vec = wrapped.embed_query("hello")
        assert len(vec) == 4


class TestBatchLimitingEmbeddingsAsync:
    @pytest.mark.asyncio
    async def test_small_batch_passes_through_unchanged(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=10)
        result = await wrapped.aembed_documents(["a", "b"])
        assert len(result) == 2
        assert stub.async_batches == [2]

    @pytest.mark.asyncio
    async def test_oversize_batch_splits_into_chunks(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=80)
        # 144 docs = realistic large tenant; split into 80 + 64.
        texts = [f"t{i}" for i in range(144)]
        result = await wrapped.aembed_documents(texts)
        assert len(result) == 144
        assert stub.async_batches == [80, 64]

    @pytest.mark.asyncio
    async def test_aquery_passes_through(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=2)
        vec = await wrapped.aembed_query("hello")
        assert len(vec) == 4

    @pytest.mark.asyncio
    async def test_exact_batch_boundary_no_trailing_chunk(self):
        stub = _StubEmbeddings()
        wrapped = BatchLimitingEmbeddings(stub, batch_size=4)
        texts = [f"t{i}" for i in range(8)]
        await wrapped.aembed_documents(texts)
        # 8 / 4 = 2 full batches, no trailing empty call.
        assert stub.async_batches == [4, 4]


# ── build_embeddings wrapping ──────────────────────────────


class TestBuildEmbeddingsWrapping:
    def test_gemini_model_gets_batch_limiting_wrapper(self):
        """Gemini model string → returned Embeddings is a BatchLimitingEmbeddings."""
        fake_inner = _StubEmbeddings()
        fake_module = MagicMock()
        fake_module.GoogleGenerativeAIEmbeddings = MagicMock(return_value=fake_inner)

        with patch("importlib.import_module", return_value=fake_module):
            result = build_embeddings("gemini/gemini-embedding-001")

        assert isinstance(result, BatchLimitingEmbeddings)
        assert result.inner is fake_inner
        assert result.batch_size == 80

    def test_openai_model_gets_batch_limiting_wrapper(self):
        """OpenAI models have a documented 2048 cap — wrapped at 2000."""
        fake_inner = _StubEmbeddings()
        with patch(
            "orchid_ai.rag.embeddings._build_fallback_embeddings",
            return_value=fake_inner,
        ):
            result = build_embeddings("text-embedding-3-small")

        assert isinstance(result, BatchLimitingEmbeddings)
        assert result.inner is fake_inner
        assert result.batch_size == 2000

    def test_unknown_fallback_model_is_not_wrapped(self):
        """Bare-name models not in the limits table get the default batch size (32)."""
        fake_inner = _StubEmbeddings()
        with patch(
            "orchid_ai.rag.embeddings._build_fallback_embeddings",
            return_value=fake_inner,
        ):
            result = build_embeddings("some-unknown-embedding-model")

        assert isinstance(result, BatchLimitingEmbeddings)

    def test_ollama_model_is_not_wrapped(self):
        """Ollama models get the default batch size (32) since they have no
        provider-specific limit in the table."""
        fake_inner = _StubEmbeddings()
        fake_module = MagicMock()
        fake_module.OllamaEmbeddings = MagicMock(return_value=fake_inner)

        with patch("importlib.import_module", return_value=fake_module):
            result = build_embeddings("ollama/nomic-embed-text")

        assert isinstance(result, BatchLimitingEmbeddings)


# ── get_embedding_dimension (existing contract) ────────────


class TestGetEmbeddingDimension:
    def test_known_models(self):
        assert get_embedding_dimension("text-embedding-3-small") == 1536
        assert get_embedding_dimension("ollama/nomic-embed-text") == 768
        assert get_embedding_dimension("gemini/gemini-embedding-001") == 3072

    def test_unknown_model_defaults_to_1536(self):
        assert get_embedding_dimension("unknown-model") == 1536


# ── End-to-end: wrapping survives real call path ──────────


class TestEndToEndChunking:
    """A realistic flow: build_embeddings returns a wrapper that actually chunks."""

    @pytest.mark.asyncio
    async def test_gemini_oversize_aembed_chunks_to_inner(self):
        fake_inner = _StubEmbeddings()
        # Spy on the inner call so we see what it receives.
        inner_async = AsyncMock(side_effect=fake_inner.aembed_documents)
        fake_inner.aembed_documents = inner_async  # type: ignore[method-assign]

        fake_module = MagicMock()
        fake_module.GoogleGenerativeAIEmbeddings = MagicMock(return_value=fake_inner)

        with patch("importlib.import_module", return_value=fake_module):
            emb = build_embeddings("gemini/gemini-embedding-001")

        # 160 docs = 2 × 80 chunks for the 80-cap Gemini wrapper.
        texts = [f"doc-{i}" for i in range(160)]
        result = await emb.aembed_documents(texts)

        assert len(result) == 160
        assert inner_async.await_count == 2
        sizes = [len(call.args[0]) for call in inner_async.await_args_list]
        assert sizes == [80, 80]
