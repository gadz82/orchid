"""
Qdrant implementation of OrchidVectorStoreRepository — multi-tenant (ADR-014).

Multi-tenancy strategy:
  - **One Qdrant collection per domain** (e.g. ``knowledge-base``, ``uploads``).
  - **Payload-based tenant filtering**: every point has a ``tenant_id`` field
    whose value is the tenant identifier from the resolved identity.
  - **Shared data** uses ``tenant_id = "__shared__"`` and is included in
    every tenant's queries automatically.
  - Retrieval always filters: ``tenant_id IN [<installation_id>, "__shared__"]``.

Hybrid-search support (ADR-025):
  - New collections are created with named dense + sparse vectors
    (``{"dense": VectorParams(...)}`` + ``sparse_vectors_config={"sparse":
    SparseVectorParams()}``).
  - Existing legacy (unnamed-vector) collections are detected on
    ``_ensure_collection`` and logged with a "recreate required for
    hybrid" migration message; reads/writes against them keep working
    in dense-only mode and ``retrieve_sparse`` raises
    :class:`NotImplementedError` so :class:`HybridRetrieval` falls
    back gracefully.
  - When a sparse encoder is injected, document writes also encode
    sparse vectors and store them under the named ``sparse`` slot.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from langchain_core.embeddings import Embeddings

from ...core.repository import Document, OrchidSearchResult, OrchidVectorStoreRepository
from ...core.scopes import SHARED_TENANT, OrchidRAGScope
from ...core.sparse import OrchidSparseEncoder, OrchidSparseVector

logger = logging.getLogger(__name__)

QDRANT_TIMEOUT = 30.0  # seconds — timeout for Qdrant operations

# Named-vector slots used by the Stage 4 hybrid schema.
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

CollectionMode = Literal["hybrid", "legacy"]


def build_qdrant_filter(scope: OrchidRAGScope) -> Filter:
    """
    Build a Qdrant ``Filter`` with ``should`` (OR) clauses covering
    every scope level visible to the caller.

    The result is a single filter that, when passed to ``client.search()``,
    returns documents from all accessible levels ranked by relevance.
    Lives next to ``QdrantRepository`` so the Qdrant client import stays
    inside this module — strategies and ``rag/scopes.py`` depend only on
    :class:`OrchidRAGScope`.
    """
    clauses: list[Filter] = []

    # 1. Root common — tenant_id = "__shared__"
    clauses.append(
        Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=SHARED_TENANT)),
            ]
        )
    )

    # 2. Tenant-level — tenant_id = T AND scope = "tenant"
    clauses.append(
        Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="scope", match=MatchValue(value="tenant")),
            ]
        )
    )

    # 3. User-common — requires user_id
    if scope.user_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="scope", match=MatchValue(value="user")),
                ]
            )
        )

    # 4. Chat-shared — requires user_id + chat_id
    if scope.user_id and scope.chat_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                    FieldCondition(key="scope", match=MatchValue(value="chat_shared")),
                ]
            )
        )

    # 5. Agent-private — requires user_id + chat_id + agent_id
    if scope.user_id and scope.chat_id and scope.agent_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=scope.agent_id)),
                    FieldCondition(key="scope", match=MatchValue(value="chat_agent")),
                ]
            )
        )

    return Filter(should=clauses)


class QdrantRepository(OrchidVectorStoreRepository):
    """
    Qdrant-backed vector store with per-tenant isolation (ADR-014).

    Each ``namespace`` maps to a Qdrant **collection** (e.g. ``knowledge-base``).
    Tenant isolation is enforced via a ``tenant_id`` payload field on every
    point, and all reads filter on ``[tenant_id, "__shared__"]``.
    """

    supports_scope_promotion = True

    def __init__(
        self,
        *,
        url: str,
        embeddings: Embeddings,
        embedding_dimension: int = 1536,
        default_tenant: str = "default",
        sparse_encoder: OrchidSparseEncoder | None = None,
        client: AsyncQdrantClient | None = None,
    ):
        # ``client`` injection point keeps unit tests fast (mocked
        # AsyncQdrantClient) without sacrificing the live URL ctor for
        # production wiring.
        self._client = client or AsyncQdrantClient(url=url)
        self._embeddings = embeddings
        self._embedding_dimension = embedding_dimension
        self._default_tenant = default_tenant
        self._sparse_encoder = sparse_encoder
        # Per-namespace schema cache: ``hybrid`` (named dense + sparse,
        # written by Stage 4+) vs ``legacy`` (unnamed dense only,
        # written by Stage 0-3 and pre-redesign).  Populated on first
        # ``_ensure_collection`` call.
        self._collection_modes: dict[str, CollectionMode] = {}

    # ── OrchidVectorReader ──────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
    ) -> list[OrchidSearchResult]:
        """
        Retrieve the *k* most relevant documents for *query* in *namespace*.

        Uses the hierarchical ``OrchidRAGScope`` to build a Qdrant filter that
        includes all scope levels visible to the caller (shared → tenant →
        user → chat_shared → chat_agent).
        """
        await self._ensure_collection(namespace)
        mode = self._collection_modes[namespace]

        query_vector = await self._embeddings.aembed_query(query)
        query_filter = self._scope_filter(scope)

        async with asyncio.timeout(QDRANT_TIMEOUT):
            # Hybrid collections need the named-vector form; legacy
            # collections still use the unnamed shape.
            query_arg: Any = NamedVector(name=_DENSE_NAME, vector=query_vector) if mode == "hybrid" else query_vector
            response = await self._client.query_points(
                collection_name=namespace,
                query=query_arg,
                query_filter=query_filter,
                limit=k,
            )

        return [
            OrchidSearchResult(
                document=Document(
                    id=str(hit.id),
                    page_content=hit.payload.get("content", "") if hit.payload else "",
                    metadata=hit.payload or {},
                ),
                score=hit.score,
            )
            for hit in response.points
        ]

    async def retrieve_sparse(
        self,
        query_sparse: OrchidSparseVector,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[OrchidSearchResult]:
        """Sparse-lane retrieval for hybrid search (ADR-025)."""
        await self._ensure_collection(namespace)
        mode = self._collection_modes[namespace]
        if mode != "hybrid":
            raise NotImplementedError(
                f"Qdrant collection '{namespace}' was created without sparse vectors — "
                "recreate the collection to enable hybrid search."
            )

        query_filter = self._scope_filter(scope)
        sparse_payload = NamedSparseVector(
            name=_SPARSE_NAME,
            vector=SparseVector(
                indices=list(query_sparse.indices),
                values=list(query_sparse.values),
            ),
        )

        async with asyncio.timeout(QDRANT_TIMEOUT):
            response = await self._client.query_points(
                collection_name=namespace,
                query=sparse_payload,
                query_filter=query_filter,
                limit=k,
            )

        return [
            OrchidSearchResult(
                document=Document(
                    id=str(hit.id),
                    page_content=hit.payload.get("content", "") if hit.payload else "",
                    metadata=hit.payload or {},
                ),
                score=hit.score,
            )
            for hit in response.points
        ]

    def _scope_filter(self, scope: OrchidRAGScope | None) -> Filter:
        if scope is not None:
            return build_qdrant_filter(scope)
        # No scope → only shared data.
        return Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchAny(any=[self._default_tenant, "__shared__"])),
            ]
        )

    async def lookup_cached_tool_results(
        self,
        namespace: str,
        scope: OrchidRAGScope,
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
        points = await self._documents_to_points(docs, namespace=namespace)
        await self._client.upsert(collection_name=namespace, points=points)

        logger.info(
            "[Qdrant] duplicated %d points in '%s' with new scope %s",
            len(docs),
            namespace,
            new_scope_fields,
        )
        return len(docs)

    # ── OrchidVectorStoreRepository — scope promotion ────────────────

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

    # ── OrchidVectorWriter ──────────────────────────────────────────

    async def index(
        self,
        documents: list[Document],
        namespace: str,
    ) -> None:
        """Index documents — creates the collection if it doesn't exist."""
        await self._ensure_collection(namespace)
        points = await self._documents_to_points(documents, namespace=namespace)
        async with asyncio.timeout(QDRANT_TIMEOUT):
            await self._client.upsert(collection_name=namespace, points=points)
        logger.info(
            "[Qdrant] indexed %d documents in '%s' (mode=%s)",
            len(documents),
            namespace,
            self._collection_modes.get(namespace, "?"),
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
        """Create or detect the Qdrant collection's schema mode.

        New collections always use the Stage 4 hybrid schema (named
        dense + sparse vectors).  Existing collections — including
        pre-Stage-4 deployments with an unnamed dense vector — are
        detected and registered as ``legacy``; reads/writes still work
        in dense-only mode but ``retrieve_sparse`` raises
        :class:`NotImplementedError` for them, prompting the operator
        to recreate the collection per the migration playbook.
        """
        if namespace in self._collection_modes:
            return

        exists = await self._client.collection_exists(namespace)
        if exists:
            self._collection_modes[namespace] = await self._detect_mode(namespace)
            return

        # Create with the named hybrid schema by default.
        await self._client.create_collection(
            collection_name=namespace,
            vectors_config={
                _DENSE_NAME: VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                _SPARSE_NAME: SparseVectorParams(),
            },
        )
        self._collection_modes[namespace] = "hybrid"
        logger.info(
            "[Qdrant] created hybrid collection '%s' (dense_dim=%d, sparse=enabled)",
            namespace,
            self._embedding_dimension,
        )

    async def _detect_mode(self, namespace: str) -> CollectionMode:
        """Inspect an existing collection to decide hybrid vs legacy."""
        info = await self._client.get_collection(namespace)
        params = info.config.params
        vectors = getattr(params, "vectors", None)
        sparse = getattr(params, "sparse_vectors", None)

        # ``vectors`` is either a dict (named) or a single VectorParams
        # (unnamed legacy schema).  The hybrid mode requires named
        # ``dense`` AND a configured sparse named slot.
        named_dense = isinstance(vectors, dict) and _DENSE_NAME in vectors
        named_sparse = isinstance(sparse, dict) and _SPARSE_NAME in sparse
        if named_dense and named_sparse:
            return "hybrid"

        logger.warning(
            "[Qdrant] collection '%s' lacks the Stage 4 named-vector schema — "
            "hybrid search disabled. Recreate the collection to enable it: "
            "stop writes, drop the collection, re-index via POST /index.",
            namespace,
        )
        return "legacy"

    # ── Internal helpers ──────────────────────────────────────

    async def _documents_to_points(
        self,
        documents: list[Document],
        *,
        namespace: str | None = None,
    ) -> list[PointStruct]:
        """Convert Documents to Qdrant PointStructs, embedding texts as needed.

        ``namespace`` selects the schema mode (hybrid vs legacy) and
        flows into the sparse encoder so per-namespace BM25 stats stay
        isolated.  When ``namespace`` is ``None`` (e.g. duplicate-with-
        new-scope path), points are written in legacy unnamed shape —
        the call sites that rely on this set ``namespace=None`` only
        when they're already inside a hybrid-aware path that re-routes
        through ``index()``.
        """
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

        # Sparse vectors — only computed when the namespace is hybrid
        # AND a sparse encoder is wired.  Legacy collections write
        # unnamed dense vectors only.
        mode: CollectionMode = self._collection_modes.get(namespace, "legacy") if namespace else "legacy"
        sparse_vectors: list[OrchidSparseVector] | None = None
        if mode == "hybrid" and self._sparse_encoder is not None:
            sparse_vectors = await self._sparse_encoder.encode_documents(
                [doc.page_content for doc in documents],
                namespace=namespace,
            )

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

            dense_vec = doc_embeddings[i] or []
            if mode == "hybrid":
                vector_payload: Any = {_DENSE_NAME: dense_vec}
                if sparse_vectors is not None:
                    sv = sparse_vectors[i]
                    if sv.indices:
                        vector_payload[_SPARSE_NAME] = SparseVector(
                            indices=list(sv.indices),
                            values=list(sv.values),
                        )
            else:
                vector_payload = dense_vec

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector_payload,
                    payload=payload,
                )
            )
        return points
