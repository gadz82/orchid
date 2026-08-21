"""
``HierarchicalIngestion`` — separate parent docstore for true hierarchical RAG.

Splits text into large parent chunks (for LLM-context hydration) and
small child chunks (for precise embedding retrieval).  When a real
:class:`OrchidDocStore` is wired, parents are written to the docstore
keyed by ``parent_id`` and child chunks carry **only** ``parent_id``
in metadata.  When ``doc_store`` is ``None`` or a no-op
:class:`NullDocStore`, the strategy degrades to today's
parent-in-metadata layout (also writes ``parent_content`` to each
child chunk) so retrieval-time hydration via
``OrchidAgent.fetch_rag_context`` still works without extra wiring.

The Stage 5+ retrieval-time hydration helper (which queries the
docstore for parent text) is out of scope here; for Stage 2 the
metadata fallback is the working hydration path.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...core.doc_store import OrchidDocStore
from ...core.ingestion import OrchidChunk, OrchidIngestionStrategy
from ...core.scopes import OrchidRAGScope, resolve_scope_level
from ..chunker import ChunkConfig

logger = logging.getLogger(__name__)


class HierarchicalIngestion(OrchidIngestionStrategy):
    """Parent / child splitter backed by an :class:`OrchidDocStore`.

    Parameters
    ----------
    config : ChunkConfig | None
        Chunk-size knobs.  ``parent_chunk_size`` defaults to
        ``chunk_size * 4`` when set to ``0`` so the strategy works
        out-of-the-box with default :class:`ChunkConfig`.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()

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

        cfg = self._config
        # ``parent_chunk_size == 0`` is the default; auto-derive a
        # parent size that's large enough to contain ~4 children.
        parent_size = cfg.parent_chunk_size or max(cfg.chunk_size * 4, cfg.chunk_size + 200)

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=cfg.parent_chunk_overlap,
            separators=[cfg.separator, "\n", " ", ""],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=[cfg.separator, "\n", " ", ""],
        )

        parent_chunks = parent_splitter.split_text(text)
        if not parent_chunks:
            return []

        # Detect the no-op fallback through the ABC's ``is_null`` marker
        # — keeps the documents/ → core/ dependency direction clean
        # (no rag/ import) per orchid/AGENTS.md.
        use_metadata_fallback = doc_store is None or getattr(doc_store, "is_null", False)
        if use_metadata_fallback:
            logger.warning(
                "[HierarchicalIngestion] No persistent OrchidDocStore wired — falling back to "
                "parent-in-metadata mode (parent text duplicated into every child chunk)."
            )

        file_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        out: list[OrchidChunk] = []

        for pi, parent_text in enumerate(parent_chunks):
            parent_id = f"parent-{file_hash}-{pi}"

            # Write to docstore unconditionally — NullDocStore swallows
            # the call so the path is the same in both branches.  The
            # metadata-fallback flag below decides what the child
            # chunks carry, not whether we attempt the put.
            if doc_store is not None:
                await doc_store.put(
                    parent_id,
                    parent_text,
                    {
                        "tenant_id": scope.tenant_id,
                        "user_id": scope.user_id,
                        "chat_id": scope.chat_id,
                        "scope": resolve_scope_level(scope),
                        "source_file": filename,
                        "parent_index": pi,
                    },
                )

            child_texts = child_splitter.split_text(parent_text) or [parent_text.strip()]
            for ci, child_text in enumerate(child_texts):
                metadata: dict[str, Any] = {
                    "tenant_id": scope.tenant_id,
                    "user_id": scope.user_id,
                    "chat_id": scope.chat_id,
                    "scope": resolve_scope_level(scope),
                    "source_file": filename,
                    "parent_id": parent_id,
                    "parent_index": pi,
                    "chunk_id": f"{parent_id}-c{ci}",
                    "chunk_index": ci,
                    "ingestion_strategy": "hierarchical",
                }
                if use_metadata_fallback:
                    metadata["parent_content"] = parent_text
                out.append(OrchidChunk(text=child_text, metadata=metadata))

        return out
