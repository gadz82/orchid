"""
Dynamic RAG — write tool data back to the vector store via the
configured ingestion strategy (ADR-024).

When a sub-agent runs an MCP or built-in tool tagged with
``inject_to_rag: true``, this module indexes the result so that:

  1. Future queries on similar topics hit cached context.
  2. The static knowledge base is progressively enriched.
  3. The supervisor's synthesis step finds richer RAG context.

Per-tool ingestion knobs (chunk size, semantic boundaries, parent
layout, …) flow in through the supplied
:class:`~orchid_ai.core.ingestion.OrchidIngestionStrategy` so the
agent's :meth:`OrchidAgentConfig.effective_rag` decision shapes the
bytes that hit the writer.

``inject_to_rag`` is safe to call on a :class:`NullVectorReader` —
it short-circuits with ``0`` when the store does not implement
:class:`OrchidVectorWriter`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from ..core.ingestion import OrchidIngestionStrategy
from ..core.repository import Document, OrchidVectorReader, OrchidVectorWriter
from .scopes import OrchidRAGScope

logger = logging.getLogger(__name__)


async def inject_to_rag(
    store: OrchidVectorReader | OrchidVectorWriter,
    *,
    tool_name: str,
    tool_result: Any,
    namespace: str,
    scope: OrchidRAGScope,
    ingestion: OrchidIngestionStrategy,
) -> int:
    """Index a single tool's result through ``ingestion`` (ADR-024).

    Per ADR-024 the dynamic-injection path delegates chunking to the
    ingestion strategy resolved by the agent's
    :meth:`OrchidAgentConfig.effective_rag` — so per-tool knobs
    (``chunk_size``, parent layout, semantic boundaries, …) apply to
    tool results just like they do to static documents.

    Parameters
    ----------
    store
        Vector store backend.  Writing only happens when the store
        implements :class:`OrchidVectorWriter`; otherwise this is a
        safe no-op.
    tool_name
        Tool that produced the result — recorded as ``source_tool``
        metadata and embedded in the chunk's deterministic ID prefix.
    tool_result
        Raw tool output.  Strings pass through; other types are
        ``json.dumps``-serialised.  Dicts carrying ``"error"`` are
        skipped.
    namespace
        Target collection — typically ``effective_rag.namespace``
        falling back to the agent's namespace.
    scope
        Hierarchical scope; flows into chunk metadata via the
        ingestion strategy.
    ingestion
        Pre-built ingestion strategy.  The agent constructs this from
        the tool's ``effective_rag.ingestion`` block via
        :func:`orchid_ai.documents.strategies.build_ingestion_strategy`.

    Returns
    -------
    int
        Number of chunks indexed (``0`` when the store doesn't
        support writing or the strategy produced no chunks).
    """
    if not isinstance(store, OrchidVectorWriter):
        logger.debug(
            "[DynamicRAG] Store does not support writing — skipping injection of '%s'",
            tool_name,
        )
        return 0

    if isinstance(tool_result, dict) and "error" in tool_result:
        return 0

    text = _serialise(tool_result)
    if not text.strip():
        return 0

    chunks = await ingestion.ingest(
        text=text,
        filename=f"tool:{tool_name}",
        scope=scope,
    )
    if not chunks:
        return 0

    documents: list[Document] = []
    for idx, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.text.encode("utf-8", errors="replace")).hexdigest()[:16]
        doc_id = chunk.metadata.get("chunk_id") or (
            f"dynamic-{tool_name}-{scope.tenant_id}-{scope.chat_id}-{idx}-{content_hash}"
        )
        metadata = {
            **chunk.metadata,
            "source_tool": tool_name,
            "dynamic": True,
            "injected_at": time.time(),
        }
        documents.append(Document(id=doc_id, page_content=chunk.text, metadata=metadata))

    try:
        await store.upsert(documents, namespace)
    except Exception as exc:
        logger.warning(
            "[DynamicRAG] Failed to inject '%s' into '%s': %s",
            tool_name,
            namespace,
            exc,
        )
        return 0

    logger.info(
        "[DynamicRAG] Injected %d chunks from '%s' into '%s' (tenant=%s, chat=%s)",
        len(documents),
        tool_name,
        namespace,
        scope.tenant_id,
        scope.chat_id,
    )
    return len(documents)


def _serialise(value: Any) -> str:
    """Coerce a tool result into a single text payload."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)
