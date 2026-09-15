"""
``RecursiveIngestion`` — flat or parent-in-metadata chunking via
``RecursiveCharacterTextSplitter``.

Wraps the existing :func:`orchid_ai.documents.chunker.chunk_text` and
:func:`parent_child_chunk_text` so the existing Stage 0 behaviour falls
out as the ``recursive`` strategy.  When ``parent_chunk_size > 0`` the
output uses today's parent-in-metadata layout — Stage 2 adds a separate
``hierarchical`` strategy that backs parent docs with an
:class:`OrchidDocStore` instead.

The strategy is constructed with a :class:`ChunkConfig` (the existing
dataclass) so the YAML loader can hand a single object across — no
behaviour-changing surface here.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ...core.ingestion import OrchidChunk, OrchidIngestionStrategy
from ...core.scopes import OrchidRAGScope, resolve_scope_level
from ..chunker import ChunkConfig, chunk_text, parent_child_chunk_text


class RecursiveIngestion(OrchidIngestionStrategy):
    """Recursive character splitter — the default ingestion strategy."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()

    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: Any | None = None,  # ignored — Stage 2's HierarchicalIngestion uses it
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        if not text.strip():
            return []

        cfg = self._config
        file_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        chunks: list[OrchidChunk] = []

        if cfg.parent_chunk_size > 0:
            pc_chunks = parent_child_chunk_text(text, cfg)
            for i, pc in enumerate(pc_chunks):
                chunk_id = f"upload-{file_hash}-p{pc.parent_index}c{pc.child_index}"
                chunks.append(
                    OrchidChunk(
                        text=pc.child_text,
                        metadata={
                            "tenant_id": scope.tenant_id,
                            "user_id": scope.user_id,
                            "chat_id": scope.chat_id,
                            "scope": resolve_scope_level(scope),
                            "source_file": filename,
                            "chunk_id": chunk_id,
                            "chunk_index": i,
                            "total_chunks": len(pc_chunks),
                            "parent_content": pc.parent_text,
                            "parent_index": pc.parent_index,
                        },
                    )
                )
        else:
            flat_chunks = chunk_text(text, cfg)
            for i, body in enumerate(flat_chunks):
                chunk_id = f"upload-{file_hash}-{i}"
                chunks.append(
                    OrchidChunk(
                        text=body,
                        metadata={
                            "tenant_id": scope.tenant_id,
                            "user_id": scope.user_id,
                            "chat_id": scope.chat_id,
                            "scope": resolve_scope_level(scope),
                            "source_file": filename,
                            "chunk_id": chunk_id,
                            "chunk_index": i,
                            "total_chunks": len(flat_chunks),
                        },
                    )
                )

        return chunks
