"""
Qdrant implementation of OrchidVectorStoreRepository — multi-tenant.

Multi-tenancy strategy:
  - **One Qdrant collection per domain** (e.g. ``knowledge-base``, ``uploads``).
  - **Payload-based tenant filtering**: every point has a ``tenant_id`` field
    whose value is the tenant identifier from the resolved identity.
  - **Shared data** uses ``tenant_id = "__shared__"`` and is included in
    every tenant's queries automatically.
  - Retrieval always filters: ``tenant_id IN [<installation_id>, "__shared__"]``.

Hybrid-search support:
  - Collections are created with named dense + sparse vectors
    (``{"dense": VectorParams(...)}`` + ``sparse_vectors_config={"sparse":
    SparseVectorParams()}``).  Collections without that schema are
    rejected on first use — the named-vector hybrid schema is the
    only supported layout.
  - When a sparse encoder is injected, document writes also encode
    sparse vectors and store them under the named ``sparse`` slot.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    NamedSparseVector,
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

# Backend-namespaced filter prefix.
_BACKEND_NS_PREFIX = "_"

# Operator → Range field mapping used by the metadata-filter translator.
_RANGE_OPERATORS = ("gte", "lte", "gt", "lt")

# Named-vector slots used by the hybrid schema.
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

# Stable v5 UUID namespace for point IDs — ensures re-indexing the same
# document replaces the point rather than creating a duplicate.
_POINT_ID_NAMESPACE = uuid.UUID("1f23d4a0-2b6e-44b9-9c5c-c2b7e3c8d1e0")


def build_metadata_filter_clauses(
    metadata_filters: dict[str, Any],
) -> tuple[list[FieldCondition], list[FieldCondition]]:
    """Translate the metadata-filter mini-language into Qdrant ``FieldCondition`` lists.

    Returns ``(must_clauses, must_not_clauses)`` so the caller can combine
    them into a single :class:`Filter` alongside any scope clauses.

    Operator handling:

      * ``"key": value`` (scalar) → ``MatchValue`` (exact match).
      * ``"key": [v1, v2, ...]`` → ``MatchAny`` (match-any).
      * ``"key": {"gte": ..., "lte": ..., "gt": ..., "lt": ...}`` →
        ``Range`` with every present bound applied.
      * ``"key": {"contains": v}`` → ``MatchValue`` against the array
        element (Qdrant's payload arrays accept this form natively).
      * ``"key": {"not": v}`` → emitted into ``must_not_clauses``.
      * Keys starting with ``_`` are backend-namespaced extras
        (e.g. ``"_qdrant"`` for raw FieldConditions an integrator
        wants to inject); skipped here since they're not part of the
        portable mini-language.

    Unknown operator dicts raise :class:`ValueError` so a YAML typo
    surfaces immediately rather than silently dropping a filter.
    """
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []

    for key, value in metadata_filters.items():
        if key.startswith(_BACKEND_NS_PREFIX):
            continue

        if isinstance(value, dict):
            range_kwargs = {op: value[op] for op in _RANGE_OPERATORS if op in value}
            if range_kwargs:
                # Datetime ranges use a different Qdrant model so the
                # client doesn't try to parse ISO-8601 strings as floats.
                # ``FieldCondition.range`` accepts both ``Range`` (numeric)
                # and ``DatetimeRange`` (ISO-8601 strings).
                range_obj: Any = (
                    DatetimeRange(**range_kwargs) if _is_datetime_range(range_kwargs) else Range(**range_kwargs)
                )
                must.append(FieldCondition(key=key, range=range_obj))
                continue
            if "contains" in value:
                must.append(FieldCondition(key=key, match=MatchValue(value=value["contains"])))
                continue
            if "not" in value:
                must_not.append(FieldCondition(key=key, match=MatchValue(value=value["not"])))
                continue
            raise ValueError(
                f"Unknown metadata filter operator(s) for {key!r}: {sorted(value)}. "
                f"Allowed: gte/lte/gt/lt/contains/not."
            )

        if isinstance(value, list):
            must.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
            continue

        # Scalar exact-match — bool / int / float / str / etc.
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return must, must_not


def infer_payload_index_types(
    metadata_filters: dict[str, Any],
) -> dict[str, str]:
    """Infer Qdrant payload-index types from a metadata-filter dict.

    Pre-condition for payload indexing: when a filter targets a field,
    the field needs a payload index for the query to be fast.
    Inference rules:

      * ``str`` → ``keyword`` (default for atomic strings — exact-match).
      * ``int`` → ``integer``.
      * ``float`` → ``float``.
      * ``bool`` → ``bool``.
      * ``list[T]`` → infer from the first non-None element.
      * ``dict`` (operator dict) → infer from the operand value type;
        ranges keyed by ISO-8601 datetime strings → ``datetime``.

    Backend-namespaced keys (``_<name>``) are skipped — they don't
    map to portable payload indexes.
    """
    out: dict[str, str] = {}

    for key, value in metadata_filters.items():
        if key.startswith(_BACKEND_NS_PREFIX):
            continue
        operand = _operator_operand(value)
        type_name = _operand_to_qdrant_type(operand)
        if type_name is not None:
            out[key] = type_name

    return out


def _operator_operand(value: Any) -> Any:
    """Return the underlying scalar an operator dict acts on."""
    if isinstance(value, dict):
        # Range: pick any bound that's present.
        for op in _RANGE_OPERATORS:
            if op in value:
                return value[op]
        if "contains" in value:
            return value["contains"]
        if "not" in value:
            return value["not"]
        return None
    if isinstance(value, list):
        return next((v for v in value if v is not None), None)
    return value


_DATETIME_HINT = ("-",)  # ISO-8601 dates always contain hyphens (e.g. 2026-05-04)


def _is_iso_datetime_string(operand: Any) -> bool:
    return (
        isinstance(operand, str)
        and any(h in operand for h in _DATETIME_HINT)
        and len(operand) >= 4
        and operand[:4].isdigit()
    )


def _is_datetime_range(range_kwargs: dict[str, Any]) -> bool:
    """Any of the range bounds being a datetime-shaped string flips the
    range to ``DatetimeRange`` so the Qdrant client doesn't try to
    parse the strings as floats."""
    for value in range_kwargs.values():
        if _is_iso_datetime_string(value):
            return True
    return False


def _operand_to_qdrant_type(operand: Any) -> str | None:
    """Map a Python value to the matching Qdrant payload-schema type."""
    if isinstance(operand, bool):
        # Check before ``int`` — Python ``bool`` is a subclass of ``int``.
        return "bool"
    if isinstance(operand, int):
        return "integer"
    if isinstance(operand, float):
        return "float"
    if isinstance(operand, str):
        # Heuristic: looks like an ISO-8601 date / datetime if it
        # contains hyphens AND parses as digits-and-separators only.
        if _is_iso_datetime_string(operand):
            return "datetime"
        return "keyword"
    return None


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
    Qdrant-backed vector store with per-tenant isolation.

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
        # Tracks which namespaces have been verified / created so the
        # ``_ensure_collection`` fast-path skips the round-trip after
        # the first call.
        self._verified_collections: set[str] = set()
        # Payload index cache keyed per ``(namespace, field, schema_type)``.
        # Once Qdrant has confirmed the index exists we skip the
        # idempotent re-create call to keep the retrieve hot path lean.
        self._payload_index_cache: dict[str, set[tuple[str, str]]] = {}

    # ── OrchidVectorReader ──────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        namespace: str,
        k: int = 5,
        scope: OrchidRAGScope | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        """
        Retrieve the *k* most relevant documents for *query* in *namespace*.

        Uses the hierarchical ``OrchidRAGScope`` to build a Qdrant filter that
        includes all scope levels visible to the caller (shared → tenant →
        user → chat_shared → chat_agent).  ``metadata_filters`` follows
        the metadata-filter operator mini-language.
        """
        await self._ensure_collection(namespace)

        query_vector = await self._embeddings.aembed_query(query)
        await self._auto_index_for_filters(namespace, metadata_filters)
        query_filter = self._compose_filter(scope, metadata_filters)

        async with asyncio.timeout(QDRANT_TIMEOUT):
            response = await self._client.query_points(
                collection_name=namespace,
                query=query_vector,
                using=_DENSE_NAME,
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
        """Sparse-lane retrieval for hybrid search."""
        await self._ensure_collection(namespace)

        await self._auto_index_for_filters(namespace, metadata_filters)
        query_filter = self._compose_filter(scope, metadata_filters)
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

    def _compose_filter(
        self,
        scope: OrchidRAGScope | None,
        metadata_filters: dict[str, Any] | None,
    ) -> Filter:
        """Combine the scope filter with metadata-filter clauses.

        The scope filter (``Filter(should=[per-level])``) becomes a
        sub-filter inside ``must`` so all levels are still considered;
        the metadata clauses join it as additional ``must`` /
        ``must_not`` entries.  When ``metadata_filters`` is empty the
        result is the bare scope filter.
        """
        scope_filter = self._scope_filter(scope)
        if not metadata_filters:
            return scope_filter

        meta_must, meta_must_not = build_metadata_filter_clauses(metadata_filters)
        return Filter(
            must=[scope_filter, *meta_must],
            must_not=meta_must_not or None,
        )

    # ── Payload indexes ─────────────────────────────

    async def ensure_payload_indexes(
        self,
        namespace: str,
        indexes: dict[str, str],
    ) -> None:
        """Idempotently create Qdrant payload indexes for the given fields.

        Called automatically from the retrieve hot path when
        ``metadata_filters`` mention fields without a known index;
        also exposed publicly so integrators (or the admin endpoint)
        can pre-create indexes from explicit YAML
        ``rag.payload_indexes`` declarations at boot time.

        Index creation is per ``(namespace, field, schema_type)``;
        repeated calls with the same triple are no-ops.  Qdrant itself
        is also idempotent — duplicate creates return success — so a
        race between two retrieve calls is harmless.
        """
        if not indexes:
            return
        await self._ensure_collection(namespace)
        cache = self._payload_index_cache.setdefault(namespace, set())
        for field, schema_type in indexes.items():
            cache_key = (field, schema_type)
            if cache_key in cache:
                continue
            try:
                await self._client.create_payload_index(
                    collection_name=namespace,
                    field_name=field,
                    field_schema=schema_type,
                )
                cache.add(cache_key)
                logger.debug(
                    "[Qdrant] payload index created '%s.%s' (%s)",
                    namespace,
                    field,
                    schema_type,
                )
            except Exception as exc:
                logger.warning(
                    "[Qdrant] payload index '%s.%s' (%s) failed: %s",
                    namespace,
                    field,
                    schema_type,
                    exc,
                )

    async def _auto_index_for_filters(
        self,
        namespace: str,
        metadata_filters: dict[str, Any] | None,
    ) -> None:
        """Infer payload-index types from the filter dict and ensure them.

        Runs only when ``metadata_filters`` is non-empty so the no-filter
        retrieve path stays untouched.
        """
        if not metadata_filters:
            return
        inferred = infer_payload_index_types(metadata_filters)
        if not inferred:
            return
        await self.ensure_payload_indexes(namespace, inferred)

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
        """Create the collection if missing; verify the hybrid schema.

        Collections always use the named-vector hybrid schema (named
        dense + sparse vectors).  An existing collection that does
        not match the schema raises :class:`RuntimeError` so the
        deployment fails loudly instead of degrading silently.
        """
        if namespace in self._verified_collections:
            return

        if await self._client.collection_exists(namespace):
            await self._assert_hybrid_schema(namespace)
            self._verified_collections.add(namespace)
            return

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
        self._verified_collections.add(namespace)
        logger.info(
            "[Qdrant] created hybrid collection '%s' (dense_dim=%d, sparse=enabled)",
            namespace,
            self._embedding_dimension,
        )

    async def _assert_hybrid_schema(self, namespace: str) -> None:
        """Reject an existing collection that lacks named dense + sparse slots."""
        info = await self._client.get_collection(namespace)
        params = info.config.params
        vectors = getattr(params, "vectors", None)
        sparse = getattr(params, "sparse_vectors", None)

        named_dense = isinstance(vectors, dict) and _DENSE_NAME in vectors
        named_sparse = isinstance(sparse, dict) and _SPARSE_NAME in sparse
        if named_dense and named_sparse:
            return

        raise RuntimeError(
            f"Qdrant collection '{namespace}' does not match the hybrid schema "
            f"(named '{_DENSE_NAME}' dense vector + named '{_SPARSE_NAME}' sparse "
            "vector).  Drop the collection and re-index."
        )

    # ── Internal helpers ──────────────────────────────────────

    async def _documents_to_points(
        self,
        documents: list[Document],
        *,
        namespace: str | None = None,
    ) -> list[PointStruct]:
        """Convert Documents to Qdrant PointStructs, embedding texts as needed.

        ``namespace`` flows into the sparse encoder so per-namespace
        BM25 stats stay isolated.
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

        # Sparse vectors — only encoded when a sparse encoder is wired.
        sparse_vectors: list[OrchidSparseVector] | None = None
        if self._sparse_encoder is not None:
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
            vector_payload: Any = {_DENSE_NAME: dense_vec}
            if sparse_vectors is not None:
                sv = sparse_vectors[i]
                if sv.indices:
                    vector_payload[_SPARSE_NAME] = SparseVector(
                        indices=list(sv.indices),
                        values=list(sv.values),
                    )

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(_POINT_ID_NAMESPACE, doc.id or doc.page_content)),
                    vector=vector_payload,
                    payload=payload,
                )
            )
        return points
