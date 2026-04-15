"""
Dynamic RAG — write tool data back to Qdrant for future retrieval (ADR-005).

When a sub-agent fetches data from MCP servers or built-in tools, this module
indexes the results into the vector store so that:
  1. Future queries on similar topics benefit from cached context.
  2. The static knowledge base is progressively enriched with live data.
  3. The Supervisor's synthesis step can find richer RAG context.

The ``inject_to_rag`` function is safe to call even when the reader
is a ``NullVectorReader`` — it simply no-ops.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from ..core.repository import Document, VectorReader, VectorWriter
from .scopes import RAGScope

logger = logging.getLogger(__name__)


async def inject_to_rag(
    reader: VectorReader,
    *,
    mcp_data: dict[str, Any],
    namespace: str,
    scope: RAGScope,
    source_tool: str = "unknown",
) -> int:
    """
    Index tool results (MCP or built-in) into the vector store.

    Parameters
    ----------
    reader : VectorReader
        The reader injected into the agent.  If it also implements
        ``VectorWriter`` (e.g. ``QdrantRepository``), data is written.
        Otherwise this is a no-op.
    mcp_data : dict
        Raw tool results (e.g. ``{"courses": "...", "enrollments": "..."}``).
    namespace : str
        Qdrant collection name (e.g. ``"learning"``, ``"notifications"``).
    scope : RAGScope
        Hierarchical scope — determines where the data lands.
    source_tool : str
        Tool name that produced this data (for metadata tracking).

    Returns
    -------
    int
        Number of documents indexed (0 if reader doesn't support writing).
    """
    if not isinstance(reader, VectorWriter):
        logger.debug(
            "[DynamicRAG] Reader does not support writing — skipping injection for '%s'",
            namespace,
        )
        return 0

    if not mcp_data or "error" in mcp_data:
        return 0

    documents = _tool_data_to_documents(mcp_data, scope, source_tool)
    if not documents:
        return 0

    try:
        await reader.upsert(documents, namespace)
        logger.info(
            "[DynamicRAG] Injected %d docs into '%s' (tenant=%s, chat=%s, tool=%s)",
            len(documents),
            namespace,
            scope.tenant_id,
            scope.chat_id,
            source_tool,
        )
        return len(documents)
    except (ConnectionError, TimeoutError, ValueError, OSError) as exc:
        logger.warning(
            "[DynamicRAG] Failed to inject into '%s': %s",
            namespace,
            exc,
        )
        return 0


def _tool_data_to_documents(
    data: dict[str, Any],
    scope: RAGScope,
    source_tool: str,
) -> list[Document]:
    """
    Convert tool results into indexable Documents.

    Each top-level key in ``data`` becomes one Document.
    Large JSON payloads are truncated to avoid embedding excessively long strings.
    """
    docs: list[Document] = []

    for key, value in data.items():
        if key == "error":
            continue

        # Serialise value to text
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, indent=2, default=str)

        # Truncate very large payloads (embeddings work best on ~512 tokens)
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        # Deterministic ID: same data → same document ID → upsert is idempotent
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_id = f"dynamic-{key}-{scope.tenant_id}-{scope.chat_id}-{content_hash}"

        docs.append(
            Document(
                id=doc_id,
                page_content=text,
                metadata={
                    "tenant_id": scope.tenant_id,
                    "user_id": scope.user_id,
                    "chat_id": scope.chat_id,
                    "scope": "chat_shared",
                    "entity_type": f"tool_{key}",
                    "source_tool": source_tool,
                    "dynamic": True,
                    "injected_at": time.time(),
                },
            )
        )

    return docs
