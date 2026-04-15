"""
Qdrant implementation of VectorStoreRepository — multi-tenant (ADR-014).

Multi-tenancy strategy:
  - **One Qdrant collection per domain** (e.g. ``learning``, ``notifications``).
  - **Payload-based tenant filtering**: every point has a ``tenant_id`` field
    whose value is the tenant identifier from the resolved identity.
  - **Shared data** uses ``tenant_id = "__shared__"`` and is included in
    every tenant's queries automatically.
  - Retrieval always filters: ``tenant_id IN [<installation_id>, "__shared__"]``.

This design maps cleanly to production (OpenSearch Serverless) where
collection-per-domain + metadata filtering is the standard pattern.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from langchain_core.embeddings import Embeddings

from ...core.repository import Document, SearchResult, VectorStoreRepository
from ..scopes import RAGScope, build_qdrant_filter

logger = logging.getLogger(__name__)

QDRANT_TIMEOUT = 30.0  # seconds — timeout for Qdrant operations


class QdrantRepository(VectorStoreRepository):
    """
    Qdrant-backed vector store with per-tenant isolation (ADR-014).

    Each ``namespace`` maps to a Qdrant **collection** (e.g. ``learning``).
    Tenant isolation is enforced via a ``tenant_id`` payload field on every
    point, and all reads filter on ``[tenant_id, "__shared__"]``.
    """

    def __init__(
        self,
        *,
        url: str,
        embeddings: Embeddings,
        embedding_dimension: int = 1536,
        default_tenant: str = "default",
    ):
        self._client = AsyncQdrantClient(url=url)
        self._embeddings = embeddings
        self._embedding_dimension = embedding_dimension
        self._default_tenant = default_tenant

    # ── VectorReader ──────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: RAGScope | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the *k* most relevant documents for *query* in *namespace*.

        Uses the hierarchical ``RAGScope`` to build a Qdrant filter that
        includes all scope levels visible to the caller (shared → tenant →
        user → chat_shared → chat_agent).
        """
        await self._ensure_collection(namespace)

        query_vector = await self._embeddings.aembed_query(query)

        # Build hierarchical filter from scope
        if scope:
            query_filter = build_qdrant_filter(scope)
        else:
            # Fallback: no scope → return only shared data
            query_filter = Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchAny(any=[self._default_tenant, "__shared__"])),
                ]
            )

        async with asyncio.timeout(QDRANT_TIMEOUT):
            response = await self._client.query_points(
                collection_name=namespace,
                query=query_vector,
                query_filter=query_filter,
                limit=k,
            )

        return [
            SearchResult(
                document=Document(
                    id=str(hit.id),
                    page_content=hit.payload.get("content", "") if hit.payload else "",
                    metadata=hit.payload or {},
                ),
                score=hit.score,
            )
            for hit in response.points
        ]

    async def lookup_cached_tool_results(
        self,
        namespace: str,
        scope: RAGScope,
        tool_name: str,
        min_injected_at: float,
    ) -> str | None:
        """Find cached dynamic tool results by metadata (no vector search)."""
        await self._ensure_collection(namespace)

        cache_filter = Filter(
            must=[
                # Scope: match tenant + chat
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                # Dynamic injection metadata
                FieldCondition(key="dynamic", match=MatchValue(value=True)),
                FieldCondition(key="source_tool", match=MatchValue(value=tool_name)),
                # TTL: only results newer than min_injected_at
                FieldCondition(key="injected_at", range=Range(gte=min_injected_at)),
            ]
        )

        async with asyncio.timeout(QDRANT_TIMEOUT):
            results, _ = await self._client.scroll(
                collection_name=namespace,
                scroll_filter=cache_filter,
                limit=1,
                with_vectors=False,
            )

        if results:
            payload = results[0].payload or {}
            content = payload.get("content", "")
            if content:
                logger.debug("[Qdrant] Cache hit for tool '%s' in '%s'", tool_name, namespace)
                return content

        return None

    # ── Scope operations (for chat sharing) ──────────────────

    async def scroll_by_filter(
        self,
        namespace: str,
        query_filter: Filter,
    ) -> list[Document]:
        """Scroll all points matching a filter — used for chat sharing."""
        await self._ensure_collection(namespace)

        documents: list[Document] = []
        offset = None

        while True:
            results, next_offset = await self._client.scroll(
                collection_name=namespace,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_vectors=True,
            )
            for point in results:
                payload = point.payload or {}
                meta = dict(payload)
                if isinstance(point.vector, list):
                    meta["_embedding"] = point.vector
                documents.append(
                    Document(
                        id=payload.get("doc_id", str(point.id)),
                        page_content=payload.get("content", ""),
                        metadata=meta,
                    )
                )
            if next_offset is None:
                break
            offset = next_offset

        return documents

    async def duplicate_with_new_scope(
        self,
        *,
        namespace: str,
        source_filter: Filter,
        new_scope_fields: dict[str, Any],
    ) -> int:
        """
        Duplicate points matching *source_filter* with updated scope metadata.

        Used to promote chat-scoped data to user-scoped (sharing).
        Returns the number of points duplicated.
        """
        docs = await self.scroll_by_filter(namespace, source_filter)
        if not docs:
            return 0

        # Update metadata on copies
        for doc in docs:
            doc.metadata.update(new_scope_fields)
            # Remove fields that should be cleared
            for key in list(doc.metadata.keys()):
                if new_scope_fields.get(key) is None and key in ("chat_id", "agent_id"):
                    doc.metadata.pop(key, None)

        # Re-index with new UUIDs (embedding already present)
        points = await self._documents_to_points(docs)
        await self._client.upsert(collection_name=namespace, points=points)

        logger.info(
            "[Qdrant] duplicated %d points in '%s' with new scope %s",
            len(docs),
            namespace,
            new_scope_fields,
        )
        return len(docs)

    # ── VectorStoreRepository — scope promotion ────────────────

    async def promote_scope(
        self,
        *,
        namespace: str,
        source_filter: Any,
        new_scope_fields: dict,
    ) -> int:
        """Promote data to a broader scope (e.g. chat → user for sharing)."""
        return await self.duplicate_with_new_scope(
            namespace=namespace,
            source_filter=source_filter,
            new_scope_fields=new_scope_fields,
        )

    # ── VectorWriter ──────────────────────────────────────────

    async def index(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Index documents — creates the collection if it doesn't exist."""
        await self._ensure_collection(namespace)
        points = await self._documents_to_points(documents)
        async with asyncio.timeout(QDRANT_TIMEOUT):
            await self._client.upsert(collection_name=namespace, points=points)
        logger.info(
            "[Qdrant] indexed %d documents in '%s'",
            len(documents),
            namespace,
        )

    async def upsert(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Insert or update documents (idempotent)."""
        await self.index(documents, namespace)

    async def delete(
        self,
        document_ids: list[str],
        namespace: str,
    ) -> None:
        """Remove documents by ID from the namespace."""
        async with asyncio.timeout(QDRANT_TIMEOUT):
            await self._client.delete(
                collection_name=namespace,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchAny(any=document_ids),
                            ),
                        ],
                    ),
                ),
            )
        logger.info(
            "[Qdrant] deleted %d documents from '%s'",
            len(document_ids),
            namespace,
        )

    # ── Collection management ─────────────────────────────────

    async def ensure_collections(self, namespaces: list[str]) -> None:
        """Pre-create collections at startup (called from lifespan)."""
        for ns in namespaces:
            await self._ensure_collection(ns)

    async def _ensure_collection(self, namespace: str) -> None:
        """Create the Qdrant collection if it doesn't exist yet."""
        exists = await self._client.collection_exists(namespace)
        if exists:
            return

        await self._client.create_collection(
            collection_name=namespace,
            vectors_config=VectorParams(
                size=self._embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "[Qdrant] created collection '%s' (dim=%d)",
            namespace,
            self._embedding_dimension,
        )

    # ── Internal helpers ──────────────────────────────────────

    async def _documents_to_points(self, documents: list[Document]) -> list[PointStruct]:
        """Convert Documents to Qdrant PointStructs, embedding texts as needed."""
        # Check which docs already have pre-computed embeddings (stored in metadata)
        texts_to_embed: list[str] = []
        embed_indices: list[int] = []
        doc_embeddings: list[list[float] | None] = []

        for i, doc in enumerate(documents):
            pre_emb = doc.metadata.get("_embedding")
            doc_embeddings.append(pre_emb)
            if pre_emb is None:
                texts_to_embed.append(doc.page_content)
                embed_indices.append(i)

        if texts_to_embed:
            computed = await self._embeddings.aembed_documents(texts_to_embed)
            for idx, emb in zip(embed_indices, computed):
                doc_embeddings[idx] = emb

        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            # Build payload: exclude internal _embedding key
            meta = {k: v for k, v in doc.metadata.items() if k != "_embedding"}
            payload = {
                "content": doc.page_content,
                "doc_id": doc.id,
                **meta,
            }
            if "tenant_id" not in payload:
                payload["tenant_id"] = self._default_tenant

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc_embeddings[i] or [],
                    payload=payload,
                )
            )
        return points
