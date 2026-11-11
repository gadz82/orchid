"""
Document ingestion pipeline — parse → strategy → upsert.

Orchestrates the flow from raw file bytes to indexed documents.  The
chunking step delegates to a configurable :class:`OrchidIngestionStrategy`;
:func:`ingest_document` plugs strategy + post-processors into the same
parse-once flow consumers use today.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.decision import OrchidDecisionPointBinding
from ..core.ingestion import OrchidChunk, OrchidChunkPostProcessor, OrchidIngestionStrategy
from ..core.repository import OrchidDocument, OrchidVectorWriter
from ..rag.scopes import OrchidRAGScope
from .decision import DocumentDescriptor, select_document_parser, select_ingestion_strategy
from .parsers import get_parser, get_parser_by_name
from .strategies import RecursiveIngestion, get_ingestion_strategy

logger = logging.getLogger(__name__)


async def extract_text(
    *,
    file_bytes: bytes,
    filename: str,
    vision_model: str = "",
    ingestion_binding: OrchidDecisionPointBinding | None = None,
) -> str:
    """
    Extract text from a file. Returned text may be empty if
    the parser cannot handle the file.

    When ``ingestion_binding`` is provided, the ``ingestion`` point may select
    a registered parser before parsing; an unknown, refused, low-confidence or
    failed answer falls back to the extension-based parser. A disabled or
    unconfigured point makes no provider call and behaves exactly as before.
    """
    if ingestion_binding is not None:
        descriptor = DocumentDescriptor.from_bytes(file_bytes=file_bytes, filename=filename)
        selection = await select_document_parser(descriptor, ingestion_binding)
        if selection.accepted and selection.value:
            parser = get_parser_by_name(selection.value, vision_model=vision_model)
            return await parser.parse(file_bytes, filename)

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
    documents_out: list[OrchidDocument] | None = None,
    ingestion_binding: OrchidDecisionPointBinding | None = None,
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

    When ``ingestion_binding`` is provided, the ``ingestion`` point may select
    the registered parser (before extraction) and the registered ingestion
    strategy (after extraction, before ``strategy.ingest``). The binding is
    optional: a disabled, unconfigured or failing point keeps the extension /
    ``ingestion`` default, and no provider call happens without a binding.
    There is no separate chunker registry — the chosen strategy owns its
    chunker. A strategy selected by the provider is constructed with that
    strategy's defaults; configured chunk knobs apply only when the selected
    class matches the caller's configured ``ingestion`` instance.

    If ``documents_out`` is provided, the list of :class:`OrchidDocument`
    objects produced by the pipeline is appended to it.  This lets
    callers (e.g. idempotent indexers) record the exact document IDs
    that were upserted without changing the return type.

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
            ingestion_binding=ingestion_binding,
        )

    if not text.strip():
        logger.warning("[Ingest] No text extracted from %s", filename)
        return 0

    logger.info("[Ingest] Using %d chars from %s", len(text), filename)

    # 2. Resolve the ingestion strategy — the decision point may override the
    #    configured default only with a registered name. Selection happens after
    #    text extraction, and the same text is passed to ``strategy.ingest``.
    strategy = ingestion or RecursiveIngestion()
    if ingestion_binding is not None:
        descriptor = DocumentDescriptor.from_bytes(file_bytes=file_bytes, filename=filename, text=text)
        selection = await select_ingestion_strategy(descriptor, ingestion_binding)
        if selection.accepted and selection.value:
            selected_strategy = get_ingestion_strategy(selection.value)
            # Keep the caller's configured knobs when the selected strategy is
            # the same class as the configured one.
            strategy = (
                ingestion if ingestion is not None and type(ingestion) is type(selected_strategy) else selected_strategy
            )

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
        OrchidDocument(
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

    if documents_out is not None:
        documents_out.extend(documents)

    return len(documents)
