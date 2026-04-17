"""
Document ingestion pipeline — parse → chunk → embed → store.

Orchestrates the full flow from raw file bytes to indexed Qdrant points.
"""

from __future__ import annotations

import hashlib
import logging

from ..core.repository import Document, VectorWriter
from ..rag.scopes import RAGScope
from .chunker import ChunkConfig, chunk_text, parent_child_chunk_text
from .parsers import get_parser

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
    scope: RAGScope,
    namespace: str = "uploads",
    writer: VectorWriter,
    chunk_config: ChunkConfig | None = None,
    vision_model: str = "",
    pre_extracted_text: str | None = None,
) -> int:
    """
    Chunk, embed, and store a document.

    If *pre_extracted_text* is provided it is used directly (avoids
    re-parsing the file — important for vision models that are slow
    and non-deterministic). Otherwise the file is parsed on the fly.

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

    # 2. Chunk — use parent/child strategy when parent_chunk_size > 0
    cfg = chunk_config or ChunkConfig()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    documents: list[Document] = []

    if cfg.parent_chunk_size > 0:
        # Parent Document Retriever: child chunks for precise embedding,
        # parent content stored in metadata for richer LLM context.
        pc_chunks = parent_child_chunk_text(text, cfg)
        if not pc_chunks:
            return 0

        logger.info(
            "[Ingest] Split %s into %d child chunks (parent_chunk_size=%d)",
            filename,
            len(pc_chunks),
            cfg.parent_chunk_size,
        )

        for i, pc in enumerate(pc_chunks):
            doc_id = f"upload-{file_hash}-p{pc.parent_index}c{pc.child_index}"
            documents.append(
                Document(
                    id=doc_id,
                    page_content=pc.child_text,
                    metadata={
                        "tenant_id": scope.tenant_id,
                        "user_id": scope.user_id,
                        "chat_id": scope.chat_id,
                        "scope": "chat_shared",
                        "source_file": filename,
                        "chunk_index": i,
                        "total_chunks": len(pc_chunks),
                        "parent_content": pc.parent_text,
                        "parent_index": pc.parent_index,
                    },
                )
            )
    else:
        # Standard chunking (existing behavior)
        chunks = chunk_text(text, cfg)
        if not chunks:
            return 0

        logger.info("[Ingest] Split %s into %d chunks", filename, len(chunks))

        for i, chunk in enumerate(chunks):
            doc_id = f"upload-{file_hash}-{i}"
            documents.append(
                Document(
                    id=doc_id,
                    page_content=chunk,
                    metadata={
                        "tenant_id": scope.tenant_id,
                        "user_id": scope.user_id,
                        "chat_id": scope.chat_id,
                        "scope": "chat_shared",
                        "source_file": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

    # 4. Index (embeddings generated lazily by the writer)
    await writer.upsert(documents, namespace)
    logger.info(
        "[Ingest] Indexed %d chunks from %s into '%s' (chat=%s)",
        len(documents),
        filename,
        namespace,
        scope.chat_id,
    )

    return len(documents)
