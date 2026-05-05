"""
``QdrantDocStore`` — :class:`OrchidDocStore` backed by a dedicated Qdrant collection.

The doc-store usage pattern is **key-value, not similarity search** —
we never query the vector space.  Qdrant still requires a vector for
every point, so the collection is created with ``size=1`` and every
point carries ``[0.0]`` as the vector value.  The trade-off is one
unused float per parent doc; a future optimisation could share the
main RAG collection if integrators want to economise on collections.

Idempotency: ``put(doc_id, ...)`` derives a deterministic UUID5 from
``doc_id`` so subsequent writes overwrite the same point.

This module is the **only** place outside ``rag/backends/qdrant.py``
that imports ``qdrant_client`` — the dependency-boundary lint
in :mod:`orchid_ai.tests.test_dependency_boundaries` enforces it.
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
    VectorParams,
)

from ...core.doc_store import OrchidDocStore

logger = logging.getLogger(__name__)

QDRANT_TIMEOUT = 30.0  # seconds — match QdrantRepository's timeout for consistency
_DOC_STORE_NAMESPACE = uuid.UUID("0eb0d1a0-1b3e-44b9-9c2c-b2b7e2c8c9d0")  # arbitrary stable v5 namespace


def _doc_id_to_uuid(doc_id: str) -> str:
    """Derive a deterministic UUID5 from a doc_id so put-twice overwrites."""
    return str(uuid.uuid5(_DOC_STORE_NAMESPACE, doc_id))


class QdrantDocStore(OrchidDocStore):
    """Qdrant-backed parent / arbitrary-document store.

    Creates its own collection (default ``__doc_store__``) on first use.
    Vectors are stored as ``[0.0]`` (size=1) — the doc store doesn't
    use similarity search, only ID lookup via ``client.retrieve``.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str = "__doc_store__",
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._client = client or AsyncQdrantClient(url=url)
        self._collection = collection_name
        self._ensured = False

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        async with asyncio.timeout(QDRANT_TIMEOUT):
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
                logger.info(
                    "[QdrantDocStore] created collection '%s' (vector size=1)",
                    self._collection,
                )
        self._ensured = True

    async def put(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        await self._ensure_collection()
        point_id = _doc_id_to_uuid(doc_id)
        payload = {"doc_id": doc_id, "content": content, **metadata}
        async with asyncio.timeout(QDRANT_TIMEOUT):
            await self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(id=point_id, vector=[0.0], payload=payload)],
            )

    async def get(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        await self._ensure_collection()
        point_id = _doc_id_to_uuid(doc_id)
        async with asyncio.timeout(QDRANT_TIMEOUT):
            records = await self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        if not records:
            return None
        payload = records[0].payload or {}
        return self._payload_to_record(payload)

    async def get_many(self, doc_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        await self._ensure_collection()
        if not doc_ids:
            return {}
        ids = [_doc_id_to_uuid(d) for d in doc_ids]
        async with asyncio.timeout(QDRANT_TIMEOUT):
            records = await self._client.retrieve(
                collection_name=self._collection,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
        out: dict[str, tuple[str, dict[str, Any]]] = {}
        for record in records:
            payload = record.payload or {}
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str):
                continue
            out[doc_id] = self._payload_to_record(payload)
        return out

    async def delete(self, doc_ids: list[str]) -> None:
        """Remove records by ``doc_id``.

        Not part of the :class:`OrchidDocStore` contract but useful for
        operators draining stale parents.  Uses payload filtering so
        callers don't need to derive the UUID5.
        """
        await self._ensure_collection()
        if not doc_ids:
            return
        async with asyncio.timeout(QDRANT_TIMEOUT):
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
                        ],
                    ),
                ),
            )

    async def find_by_metadata(self, *, key: str, value: Any) -> list[tuple[str, str, dict[str, Any]]]:
        """Scroll the collection for points whose payload key equals ``value``.

        Returns ``[(doc_id, content, metadata)]``.  Convenience helper
        for operators auditing the store; not part of the ABC.
        """
        await self._ensure_collection()
        async with asyncio.timeout(QDRANT_TIMEOUT):
            records, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key=key, match=MatchValue(value=value))],
                ),
                limit=1024,
                with_vectors=False,
            )
        out: list[tuple[str, str, dict[str, Any]]] = []
        for record in records:
            payload = record.payload or {}
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str):
                continue
            content, metadata = self._payload_to_record(payload)
            out.append((doc_id, content, metadata))
        return out

    @staticmethod
    def _payload_to_record(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        content = str(payload.get("content", ""))
        metadata = {k: v for k, v in payload.items() if k not in ("doc_id", "content")}
        return content, metadata
