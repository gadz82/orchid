"""
``HeaderedIngestion`` — recursive chunking + contextual headers.

Convenience strategy that wires :class:`RecursiveIngestion` with the
:class:`ContextualHeaderPostProcessor` so users opting into
``ingestion.strategy: headered`` from YAML get the prepended
``# {title}\\n## {section}\\n\\n`` header without separately listing the
post-processor.

Equivalent to::

    ingestion:
      strategy: recursive
      post_processors: [contextual_headers]

The two-pronged shape mirrors the ADR-022 §"Concrete strategies"
sketch: ``headered`` is "recursive + ContextualHeaderPostProcessor".
"""

from __future__ import annotations

from typing import Any

from ...core.doc_store import OrchidDocStore
from ...core.ingestion import OrchidChunk, OrchidIngestionStrategy
from ...core.scopes import OrchidRAGScope
from ..chunker import ChunkConfig
from ..post_processors.contextual_headers import ContextualHeaderPostProcessor
from .recursive import RecursiveIngestion


class HeaderedIngestion(OrchidIngestionStrategy):
    """:class:`RecursiveIngestion` followed by :class:`ContextualHeaderPostProcessor`."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._inner = RecursiveIngestion(config)
        self._post = ContextualHeaderPostProcessor()

    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: OrchidDocStore | None = None,
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        chunks = await self._inner.ingest(
            text=text, filename=filename, scope=scope, doc_store=doc_store, embeddings=embeddings
        )
        return await self._post.process(chunks, text=text, filename=filename, chat_model=None)
