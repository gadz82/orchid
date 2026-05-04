"""
Document ingestion pipeline — parse → strategy → upsert.

Orchestrates the flow from raw file bytes to indexed documents.  The
chunking step delegates to a configurable :class:`OrchidIngestionStrategy`
(ADR-022); :func:`ingest_document` plugs strategy + post-processors into
the same parse-once flow consumers use today.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.ingestion import OrchidChunk, OrchidChunkPostProcessor, OrchidIngestionStrategy
from ..core.repository import Document, OrchidVectorWriter
from ..rag.scopes import OrchidRAGScope
from .parsers import get_parser
from .strategies import RecursiveIngestion

logger = logging.getLogger(__name__)


async def extract_text(
    *,
    file_bytes: bytes,
    filename: str,
    vision_model: str = "",
) -> str:
    """
    Extract text from a file. Returned text may be empty if
    the parser cannot handle the file.
    """
    parser = get_parser(filename, vision_model=vision_model)
    return await parser.parse(file_bytes, filename)


async def ingest_document(
    *,
    file_bytes: bytes,
    filename: str,
    scope: OrchidRAGScope,
    namespace: str = "uploads",
    writer: OrchidVectorWriter,
    ingestion: OrchidIngestionStrategy | None = None,
    post_processors: list[OrchidChunkPostProcessor] | None = None,
    doc_store: Any | None = None,
    graph_store: Any | None = None,
    embeddings: Any | None = None,
    chat_model: Any | None = None,
    schema: dict[str, Any] | None = None,
    vision_model: str = "",
    pre_extracted_text: str | None = None,
) -> int:
    """
    Parse, chunk, and upsert a document.

    The chunking step delegates to ``ingestion`` (defaults to
    :class:`RecursiveIngestion` with default :class:`ChunkConfig` —
    matching today's flat behaviour).  ``post_processors`` run in order
    after the strategy splits the text.

    Stage 2+ side-channel resources flow through to the strategy and
    post-processors:

      * ``doc_store`` — :class:`HierarchicalIngestion` writes parents.
      * ``embeddings`` — :class:`SemanticIngestion` measures similarity.
      * ``graph_store`` + ``chat_model`` — :class:`EntityExtractionPostProcessor`
        extracts entities and writes them.  ``schema`` carries an
        optional per-namespace constraint dict (``entity_types``,
        ``relations``).

    If ``pre_extracted_text`` is provided it is used directly (avoids
    re-parsing the file — important for vision models).  Otherwise the
    file is parsed on the fly.

    Returns the number of chunks indexed.
    """
    # 1. Obtain text — reuse if already extracted
    if pre_extracted_text is not None:
        text = pre_extracted_text
    else:
        text = await extract_text(
            file_bytes=file_bytes,
            filename=filename,
            vision_model=vision_model,
        )

    if not text.strip():
        logger.warning("[Ingest] No text extracted from %s", filename)
        return 0

    logger.info("[Ingest] Using %d chars from %s", len(text), filename)

    # 2. Run the configured ingestion strategy
    strategy = ingestion or RecursiveIngestion()
    chunks: list[OrchidChunk] = await strategy.ingest(
        text=text,
        filename=filename,
        scope=scope,
        doc_store=doc_store,
        embeddings=embeddings,
    )
    if not chunks:
        return 0

    # 3. Run post-processors in order (Stage 2+ feature; empty by default).
    #    Side-channel resources (chat_model, graph_store, scope, schema)
    #    flow through so processors like EntityExtractionPostProcessor
    #    can read them without an instance-level closure.
    for proc in post_processors or []:
        chunks = await proc.process(
            chunks,
            text=text,
            filename=filename,
            chat_model=chat_model,
            graph_store=graph_store,
            scope=scope,
            schema=schema,
        )

    logger.info("[Ingest] Strategy %s produced %d chunks for %s", type(strategy).__name__, len(chunks), filename)

    # 4. Convert to Documents and upsert (embeddings generated lazily by writer)
    documents = [
        Document(
            id=c.metadata.get("chunk_id") or f"{filename}-{i}",
            page_content=c.text,
            metadata=c.metadata,
        )
        for i, c in enumerate(chunks)
    ]

    await writer.upsert(documents, namespace)
    logger.info(
        "[Ingest] Indexed %d chunks from %s into '%s' (chat=%s)",
        len(documents),
        filename,
        namespace,
        scope.chat_id,
    )

    return len(documents)
