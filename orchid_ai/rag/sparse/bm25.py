"""
BM25 sparse encoder.

In-process implementation of BM25-Okapi with:

  * **Lazy IDF table per namespace** — first call to
    :meth:`encode_documents` for a namespace bootstraps the document-
    frequency table; later calls update it incrementally.  Each
    ``namespace`` argument keys its own state so a multi-tenant /
    multi-collection app doesn't cross-pollute statistics.
  * **Drift-based refresh** — the average document length used by the
    BM25 saturation factor is recomputed when the namespace's
    ``n_documents`` grows by ≥ 20% since the last refresh.  Catches
    bulk-load skew without paying the cost on every document.
  * **Deterministic token → vocab index** — MD5-based hash truncated
    to 32 bits modulo ``vocab_size``.  Stable across processes so
    Qdrant points indexed in one run remain searchable in the next.

The encoder writes BM25-weighted sparse vectors for documents and
IDF-only sparse vectors for queries (the textbook
:math:`tf \\cdot idf` document weighting / :math:`idf` query weighting
combo of the BM25-Okapi paper).  Tokens absent from the corpus
contribute nothing to the query vector.

Stop-word and tokenisation choices favour English prose; integrators
serving non-Latin corpora register a custom encoder via
:func:`orchid_ai.rag.sparse.register_sparse_encoder`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ...core.sparse import OrchidSparseEncoder, OrchidSparseVector

# Standard BM25-Okapi parameters (Robertson et al., 1995).
_K1 = 1.5
_B = 0.75

# Drift threshold for avgdl refresh — beyond ``DEFAULT_DRIFT`` growth
# in n_documents we recompute the running average.  Default ±20%.
_DEFAULT_DRIFT = 0.2

_DEFAULT_VOCAB_SIZE = 100_000


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")

#: Minimal English stop-word set.  Kept short on purpose — over-aggressive
#: stop-word removal hurts recall on technical corpora (e.g. dropping
#: ``of`` from "Bill of Materials").  Integrators with stricter needs
#: subclass and override :meth:`BM25Encoder._tokenize`.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


def _tokenize(text: str) -> list[str]:
    """Module-level tokeniser shared by :class:`BM25Encoder` and tests.

    Lowercase + alphanumeric extraction + stop-word removal.
    """
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if tok.lower() not in _STOP_WORDS]


@dataclass
class _NamespaceState:
    """Per-namespace BM25 statistics (lazy + drift-refreshed)."""

    df: dict[str, int] = field(default_factory=dict)
    n_documents: int = 0
    total_doc_length: int = 0
    avgdl: float = 0.0
    last_refresh_n: int = 0


def _hash_token(token: str, vocab_size: int) -> int:
    """Deterministic 32-bit MD5 → vocabulary slot.

    Stable across Python processes — Qdrant points indexed under one
    run remain searchable under the next.  MD5's not cryptographic
    here; collisions are managed by the BM25 weight magnitude itself
    (rare-token weights remain dominant even if a few common tokens
    happen to share a slot).
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % vocab_size


class BM25Encoder(OrchidSparseEncoder):
    """In-process BM25-Okapi sparse encoder.

    Parameters
    ----------
    vocab_size : int
        Size of the hash-vocab slot space.  Larger reduces collisions
        at the cost of the sparse-vector index range.  ``100_000`` is
        the default and works well for corpora up to ~1M documents.
    drift_threshold : float
        Fractional change in ``n_documents`` since the last avgdl
        refresh that triggers recomputation.  Default ``0.2``.
    """

    def __init__(
        self,
        *,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        drift_threshold: float = _DEFAULT_DRIFT,
    ) -> None:
        if vocab_size < 1024:
            raise ValueError(f"vocab_size must be >= 1024; got {vocab_size}")
        if not 0.0 < drift_threshold < 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1); got {drift_threshold}")
        self._vocab_size = vocab_size
        self._drift_threshold = drift_threshold
        self._states: dict[str, _NamespaceState] = {}

    # ── ABC contract ──────────────────────────────────────────

    async def encode_documents(
        self,
        texts: list[str],
        namespace: str | None = None,
    ) -> list[OrchidSparseVector]:
        state = self._state(namespace)
        out: list[OrchidSparseVector] = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                out.append(OrchidSparseVector(indices=[], values=[]))
                continue
            self._update_corpus_stats(state, tokens)
            out.append(self._weight_doc(state, tokens))
        return out

    async def encode_query(
        self,
        text: str,
        namespace: str | None = None,
    ) -> OrchidSparseVector:
        state = self._state(namespace)
        tokens = self._tokenize(text)
        if not tokens:
            return OrchidSparseVector(indices=[], values=[])
        return self._weight_query(state, tokens)

    # ── Hooks for subclasses ──────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase + alphanumeric extraction + stop-word removal.

        Override in a subclass for custom tokenisation (stemming,
        non-Latin scripts, domain-specific dictionaries).
        """
        return _tokenize(text)

    # ── Internal helpers ──────────────────────────────────────

    def _state(self, namespace: str | None) -> _NamespaceState:
        ns = namespace or "default"
        state = self._states.get(ns)
        if state is None:
            state = _NamespaceState()
            self._states[ns] = state
        return state

    def _update_corpus_stats(self, state: _NamespaceState, tokens: list[str]) -> None:
        """Update DF table, document count, total length; refresh avgdl on drift."""
        for tok in set(tokens):
            state.df[tok] = state.df.get(tok, 0) + 1
        state.n_documents += 1
        state.total_doc_length += len(tokens)
        # First-document bootstrap.
        if state.last_refresh_n == 0:
            state.avgdl = state.total_doc_length / state.n_documents
            state.last_refresh_n = state.n_documents
            return
        drift = abs(state.n_documents - state.last_refresh_n) / max(state.last_refresh_n, 1)
        if drift >= self._drift_threshold:
            state.avgdl = state.total_doc_length / state.n_documents
            state.last_refresh_n = state.n_documents

    def _weight_doc(self, state: _NamespaceState, tokens: list[str]) -> OrchidSparseVector:
        """BM25-Okapi document weighting: ``idf * tf * (k1+1) / (tf + k1 * (1-b + b*|d|/avgdl))``."""
        tf = Counter(tokens)
        n = state.n_documents
        avgdl = state.avgdl or 1.0  # bootstrap path: pretend avgdl=1 to avoid div-by-zero
        doc_len = len(tokens)

        indices: list[int] = []
        values: list[float] = []
        for tok, freq in tf.items():
            df = state.df.get(tok, 0) or 1  # min 1 — we just inserted this doc
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            denom = freq + _K1 * (1.0 - _B + _B * doc_len / avgdl)
            tf_weight = freq * (_K1 + 1.0) / max(denom, 1e-9)
            weight = idf * tf_weight
            if weight > 0:
                indices.append(_hash_token(tok, self._vocab_size))
                values.append(weight)
        return OrchidSparseVector(indices=indices, values=values)

    def _weight_query(self, state: _NamespaceState, tokens: list[str]) -> OrchidSparseVector:
        """Query weighting: IDF-only (canonical BM25 query side)."""
        # Empty corpus path: emit uniform 1.0 weights so the encoder is
        # still usable before any documents have been seen (e.g. tests).
        if state.n_documents == 0:
            indices: list[int] = []
            values: list[float] = []
            for tok in dict.fromkeys(tokens):  # preserve order, dedupe
                indices.append(_hash_token(tok, self._vocab_size))
                values.append(1.0)
            return OrchidSparseVector(indices=indices, values=values)

        n = state.n_documents
        indices = []
        values = []
        for tok in dict.fromkeys(tokens):
            df = state.df.get(tok, 0)
            if df == 0:
                continue  # OOV tokens contribute nothing
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            if idf > 0:
                indices.append(_hash_token(tok, self._vocab_size))
                values.append(idf)
        return OrchidSparseVector(indices=indices, values=values)
