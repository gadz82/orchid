"""
``SemanticIngestion`` — embedding-driven boundary detection.

Splits text into sentences, embeds each sentence via the embedder
injected by the pipeline (no concrete embedder import — the
:class:`langchain_core.embeddings.Embeddings` ABC is duck-typed
through ``Any``), then walks consecutive sentences and cuts wherever
the cosine similarity to the next drops below a threshold derived from
a percentile of the observed similarity distribution.

When the embedder is ``None`` or the text is below ``min_chunk_chars``,
the strategy degrades to :class:`RecursiveIngestion` — semantic
chunking on tiny inputs costs an LLM call for no recall gain.

Boundaries are computed locally per document, so the strategy is
stateless across calls.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any

from ...core.doc_store import OrchidDocStore
from ...core.ingestion import OrchidChunk, OrchidIngestionStrategy
from ...core.scopes import OrchidRAGScope
from .recursive import RecursiveIngestion

logger = logging.getLogger(__name__)


# A sentence ends with one of ``.``/``!``/``?`` followed by whitespace
# and an uppercase letter, OR by a paragraph break.  The regex is
# intentionally simple — semantic boundaries don't need a full
# tokenizer for Stage 2, and adding NLTK / spaCy would balloon the
# install footprint.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\n+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences via a simple punctuation rule."""
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-percentile of ``values`` (linear interpolation)."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


class SemanticIngestion(OrchidIngestionStrategy):
    """Sentence-similarity-driven chunker.

    Parameters
    ----------
    breakpoint_percentile : float
        Cut where consecutive cosine similarity falls **at or below**
        the value at this percentile of the observed distribution.
        Lower → more boundaries → smaller chunks.  Default ``5.0`` (the
        bottom 5% of similarities become breakpoints).
    min_chunk_chars : int
        Below this size, defer to :class:`RecursiveIngestion` and skip
        the embedding cost entirely.
    fallback_chunk_size : int
        ``chunk_size`` passed to the recursive fallback when the input
        is too short.
    """

    def __init__(
        self,
        *,
        breakpoint_percentile: float = 5.0,
        min_chunk_chars: int = 500,
        fallback_chunk_size: int = 1000,
    ) -> None:
        if not 0.0 < breakpoint_percentile < 100.0:
            raise ValueError(f"breakpoint_percentile must be in (0, 100); got {breakpoint_percentile}")
        self._breakpoint_percentile = breakpoint_percentile
        self._min_chunk_chars = min_chunk_chars
        self._fallback_chunk_size = fallback_chunk_size

    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: OrchidDocStore | None = None,
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        if not text.strip():
            return []

        if embeddings is None:
            logger.warning("[SemanticIngestion] No embedder injected — falling back to RecursiveIngestion.")
            return await self._fallback().ingest(text=text, filename=filename, scope=scope, doc_store=doc_store)

        if len(text) < self._min_chunk_chars:
            logger.debug(
                "[SemanticIngestion] Text length %d < min_chunk_chars %d — using recursive fallback.",
                len(text),
                self._min_chunk_chars,
            )
            return await self._fallback().ingest(text=text, filename=filename, scope=scope, doc_store=doc_store)

        sentences = _split_sentences(text)
        if len(sentences) < 2:
            return await self._fallback().ingest(text=text, filename=filename, scope=scope, doc_store=doc_store)

        try:
            sentence_vectors = await embeddings.aembed_documents(sentences)
        except Exception as exc:
            logger.warning("[SemanticIngestion] Embedder failed (%s); falling back to recursive.", exc)
            return await self._fallback().ingest(text=text, filename=filename, scope=scope, doc_store=doc_store)

        # Consecutive cosine similarities — list length is len(sentences) - 1.
        similarities = [_cosine(sentence_vectors[i], sentence_vectors[i + 1]) for i in range(len(sentences) - 1)]
        threshold = _percentile(similarities, self._breakpoint_percentile)

        # Group sentences by walking the similarity list; cut at every
        # consecutive similarity at-or-below the threshold.
        groups: list[list[str]] = [[sentences[0]]]
        for i, sim in enumerate(similarities):
            if sim <= threshold:
                groups.append([sentences[i + 1]])
            else:
                groups[-1].append(sentences[i + 1])

        return self._groups_to_chunks(groups, text=text, filename=filename, scope=scope)

    def _fallback(self) -> RecursiveIngestion:
        from ..chunker import ChunkConfig

        return RecursiveIngestion(ChunkConfig(chunk_size=self._fallback_chunk_size))

    @staticmethod
    def _groups_to_chunks(
        groups: list[list[str]],
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
    ) -> list[OrchidChunk]:
        file_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        out: list[OrchidChunk] = []
        for i, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue
            out.append(
                OrchidChunk(
                    text=chunk_text,
                    metadata={
                        "tenant_id": scope.tenant_id,
                        "user_id": scope.user_id,
                        "chat_id": scope.chat_id,
                        "scope": "chat_shared",
                        "source_file": filename,
                        "chunk_id": f"semantic-{file_hash}-{i}",
                        "chunk_index": i,
                        "total_chunks": len(groups),
                        "ingestion_strategy": "semantic",
                    },
                )
            )
        return out
