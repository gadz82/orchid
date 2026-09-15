"""``FrontMatterIngestion`` — parse YAML front-matter and inject it into chunk metadata.

Wraps an inner strategy (default ``HeaderedIngestion``).  For Markdown
files with YAML front-matter, the front-matter is stripped from the body
and merged into every chunk's metadata.  A configurable ``id_field`` can
be used to derive stable chunk IDs from a front-matter value such as
``page_id`` or ``article_id``.
"""

from __future__ import annotations

import re
from typing import Any

from ...config.frontmatter import parse_frontmatter
from ...core.doc_store import OrchidDocStore
from ...core.ingestion import OrchidChunk, OrchidIngestionStrategy
from ...core.scopes import OrchidRAGScope
from .headered import HeaderedIngestion


class FrontMatterIngestion(OrchidIngestionStrategy):
    """Ingestion strategy that parses YAML front-matter from Markdown text.

    Parameters
    ----------
    inner:
        Strategy used to chunk the body after front-matter is removed.
        Defaults to ``HeaderedIngestion()``.
    id_field:
        Optional front-matter field whose value is used to build stable
        chunk IDs.  If the field is missing, the inner strategy's IDs are
        kept.
    """

    def __init__(
        self,
        *,
        inner: OrchidIngestionStrategy | None = None,
        id_field: str = "",
    ) -> None:
        self._inner = inner or HeaderedIngestion()
        self._id_field = id_field.strip()

    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: OrchidDocStore | None = None,
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        front_matter, body = parse_frontmatter(text)
        source_id, source_from_field = self._extract_source_id(front_matter, filename)

        chunks = await self._inner.ingest(
            text=body,
            filename=filename,
            scope=scope,
            doc_store=doc_store,
            embeddings=embeddings,
        )

        return [
            self._augment_chunk(chunk, front_matter, source_id, source_from_field, filename, index=i)
            for i, chunk in enumerate(chunks)
        ]

    def _extract_source_id(self, front_matter: dict[str, Any], filename: str) -> tuple[str, bool]:
        if self._id_field:
            value = front_matter.get(self._id_field)
            if isinstance(value, str) and value:
                return self._sanitize_id(value), True
            if isinstance(value, int):
                return self._sanitize_id(str(value)), True
        return self._sanitize_id(filename), False

    def _augment_chunk(
        self,
        chunk: OrchidChunk,
        front_matter: dict[str, Any],
        source_id: str,
        source_from_field: bool,
        filename: str,
        index: int,
    ) -> OrchidChunk:
        metadata = dict(chunk.metadata)
        metadata["source_file"] = filename

        for key, value in front_matter.items():
            meta_key = f"frontmatter_{key}"
            if meta_key not in metadata:
                metadata[meta_key] = value

        if self._id_field and source_from_field and source_id:
            metadata["chunk_id"] = f"src-{source_id}-{index}"
            metadata["source_id"] = source_id

        return OrchidChunk(text=chunk.text, metadata=metadata)

    @staticmethod
    def _sanitize_id(value: str) -> str:
        """Make a value safe to embed in a chunk ID."""
        cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", value)
        return cleaned.strip("_") or "unknown"
