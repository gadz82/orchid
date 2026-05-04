"""Tests for ``BM25Encoder`` (ADR-025)."""

from __future__ import annotations

import math

import pytest

from orchid_ai.rag.sparse.bm25 import (
    _DEFAULT_VOCAB_SIZE,
    BM25Encoder,
    _hash_token,
    _tokenize,
)


class TestTokenisation:
    def test_lowercases_and_extracts_alphanumeric(self):
        assert _tokenize("Hello, World!") == ["hello", "world"]

    def test_drops_stop_words(self):
        # "the" / "of" / "is" are stop words; "data" / "structures" stay.
        assert _tokenize("The of is data structures") == ["data", "structures"]

    def test_keeps_underscores_and_hyphens_inside_tokens(self):
        # Identifier-like tokens should survive intact.
        out = _tokenize("snake_case_var camelCase kebab-case")
        assert "snake_case_var" in out
        assert "camelcase" in out
        assert "kebab-case" in out

    def test_empty_text_returns_empty_list(self):
        assert _tokenize("") == []
        assert _tokenize("   \n\t") == []


class TestHashStability:
    def test_same_token_same_index(self):
        a = _hash_token("orchid", _DEFAULT_VOCAB_SIZE)
        b = _hash_token("orchid", _DEFAULT_VOCAB_SIZE)
        assert a == b

    def test_index_within_vocab(self):
        for tok in ["a", "the", "abc123", "supercalifragilistic"]:
            idx = _hash_token(tok, _DEFAULT_VOCAB_SIZE)
            assert 0 <= idx < _DEFAULT_VOCAB_SIZE


class TestEncodeDocuments:
    @pytest.mark.asyncio
    async def test_returns_one_vector_per_input(self):
        encoder = BM25Encoder()
        out = await encoder.encode_documents(["alpha doc", "beta doc", "gamma doc"])
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_sparse_vector(self):
        encoder = BM25Encoder()
        out = await encoder.encode_documents([""])
        assert out == [out[0].__class__(indices=[], values=[])]

    @pytest.mark.asyncio
    async def test_rare_token_outweighs_common_token(self):
        """Documents only mention 'common' once; one mentions a rare 'sentinel'.

        Query for 'sentinel' should produce a higher IDF weight than for
        'common' because rarer terms carry more information.
        """
        encoder = BM25Encoder()
        # 9 docs with the common token, 1 with both common + sentinel.
        corpus = ["common word"] * 9 + ["common word sentinel"]
        await encoder.encode_documents(corpus)

        sentinel_q = await encoder.encode_query("sentinel")
        common_q = await encoder.encode_query("common")
        assert sentinel_q.values
        assert common_q.values
        assert max(sentinel_q.values) > max(common_q.values)

    @pytest.mark.asyncio
    async def test_idf_table_grows_per_namespace(self):
        encoder = BM25Encoder()
        await encoder.encode_documents(["alpha beta"], namespace="ns_a")
        await encoder.encode_documents(["gamma delta"], namespace="ns_b")
        # IDF tables are namespace-isolated.
        state_a = encoder._states["ns_a"]
        state_b = encoder._states["ns_b"]
        assert "alpha" in state_a.df and "alpha" not in state_b.df
        assert "gamma" in state_b.df and "gamma" not in state_a.df

    @pytest.mark.asyncio
    async def test_drift_refresh_updates_avgdl(self):
        """After ≥20% growth in n_documents, avgdl is recomputed."""
        encoder = BM25Encoder(drift_threshold=0.2)
        # 5 docs of length 2 → avgdl 2.0; bootstrap fires at first doc.
        await encoder.encode_documents(["one word"] * 5)
        state = encoder._states["default"]
        first_avgdl = state.avgdl
        first_refresh_n = state.last_refresh_n

        # Add 1 more doc of length 4 → avgdl drift > 20% from refresh n=5
        await encoder.encode_documents(["a b c d"])
        # Refresh point should have advanced.
        assert state.last_refresh_n > first_refresh_n
        # New avgdl should reflect the new (longer) doc.
        assert state.avgdl > first_avgdl


class TestEncodeQuery:
    @pytest.mark.asyncio
    async def test_cold_start_uniform_weights(self):
        """No corpus seen → query falls back to uniform 1.0 weights."""
        encoder = BM25Encoder()
        q = await encoder.encode_query("alpha beta")
        assert q.values
        assert all(v == 1.0 for v in q.values)

    @pytest.mark.asyncio
    async def test_oov_tokens_dropped(self):
        encoder = BM25Encoder()
        await encoder.encode_documents(["alpha beta gamma"])
        # "delta" is OOV — not in the corpus.
        q = await encoder.encode_query("delta")
        assert q.indices == []
        assert q.values == []

    @pytest.mark.asyncio
    async def test_idf_formula_matches_robertson(self):
        """Query weight equals log((N - df + 0.5) / (df + 0.5) + 1)."""
        encoder = BM25Encoder()
        # 10 docs, "alpha" in 3, "beta" in 1.
        corpus = ["alpha beta"] + ["alpha"] * 2 + ["unrelated"] * 7
        await encoder.encode_documents(corpus)
        q = await encoder.encode_query("alpha beta")
        # Two distinct tokens → two indices/values.
        assert len(q.indices) == 2

        # Recompute the expected weights manually.
        n = 10
        df_alpha = 3
        df_beta = 1
        idf_alpha = math.log((n - df_alpha + 0.5) / (df_alpha + 0.5) + 1.0)
        idf_beta = math.log((n - df_beta + 0.5) / (df_beta + 0.5) + 1.0)
        # The vector is index-sorted; we don't know which slot is which
        # without re-hashing — so just assert both IDFs are present.
        sorted_values = sorted(q.values)
        assert sorted_values == sorted([idf_alpha, idf_beta])

    @pytest.mark.asyncio
    async def test_namespace_isolated(self):
        """encode_query reads only the namespace's IDF state."""
        encoder = BM25Encoder()
        await encoder.encode_documents(["alpha beta"] * 3, namespace="A")
        q_a = await encoder.encode_query("alpha", namespace="A")
        # Namespace B has no corpus → cold-start uniform fallback.
        q_b = await encoder.encode_query("alpha", namespace="B")
        assert q_b.values == [1.0]  # cold-start
        # Namespace A produces non-trivial IDF (not 1.0).
        assert q_a.values != [1.0]


class TestValidation:
    def test_zero_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            BM25Encoder(vocab_size=0)

    def test_invalid_drift_raises(self):
        with pytest.raises(ValueError, match="drift_threshold"):
            BM25Encoder(drift_threshold=0.0)
        with pytest.raises(ValueError, match="drift_threshold"):
            BM25Encoder(drift_threshold=1.5)
