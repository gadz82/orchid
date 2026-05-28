"""
Static RAG indexer — seed the vector store with test data.

This module provides a generic ``StaticIndexer`` that can be extended
by consumers to add domain-specific seed data. The framework itself
no longer ships any hardcoded seed documents — those cab belong in consumer
projects

Consumers register their namespaces and documents, then call ``index_all()``
to seed the vector store at startup.

Usage (startup hook):
    from orchid_ai.rag.indexer import StaticIndexer

    indexer = StaticIndexer(writer=reader)
    indexer.register_namespace("knowledge-base", shared_docs, tenant_docs_fn)
    await indexer.index_all(tenant_key="12345")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ..core.repository import OrchidDocument, OrchidVectorWriter

logger = logging.getLogger(__name__)

SHARED_TENANT = "__shared__"


class StaticIndexer:
    """
    Generic static data indexer for the vector store.

    Consumers register namespaces with shared and tenant-specific documents,
    then call ``index_all()`` or ``index_shared_only()`` to seed the store.
    """

    def __init__(self, writer: OrchidVectorWriter):
        self._writer = writer
        self._namespaces: dict[str, _NamespaceData] = {}

    def register_namespace(
        self,
        namespace: str,
        shared_docs: list[OrchidDocument],
        tenant_docs_fn: Callable[[str], list[OrchidDocument]] | None = None,
    ) -> None:
        """
        Register a namespace with its seed data.

        Parameters
        ----------
        namespace : str
            The vector store collection name (e.g. ``"knowledge-base"``).
        shared_docs : list[OrchidDocument]
            Documents visible to all tenants (``tenant_id = "__shared__"``).
        tenant_docs_fn : callable, optional
            A function that takes ``tenant_key`` and returns tenant-specific
            documents. Called during ``index_all()`` for each tenant.
        """
        self._namespaces[namespace] = _NamespaceData(
            shared_docs=shared_docs,
            tenant_docs_fn=tenant_docs_fn,
        )
        logger.debug("[Indexer] Registered namespace '%s' (%d shared docs)", namespace, len(shared_docs))

    async def index_all(self, tenant_key: str = "default") -> dict[str, int]:
        """
        Index all registered data (shared + tenant-specific) for the given tenant.

        Returns a dict of ``{namespace: document_count}`` indexed.
        """
        counts: dict[str, int] = {}

        for namespace, data in self._namespaces.items():
            docs = list(data.shared_docs)
            if data.tenant_docs_fn:
                docs.extend(data.tenant_docs_fn(tenant_key))

            await self._writer.index(docs, namespace)
            counts[namespace] = len(docs)
            logger.info(
                "[Indexer] Indexed %d docs in '%s' (tenant=%s + shared)",
                len(docs),
                namespace,
                tenant_key,
            )

        return counts

    async def index_shared_only(self) -> dict[str, int]:
        """Index only shared (cross-tenant) documents for all namespaces."""
        counts: dict[str, int] = {}

        for namespace, data in self._namespaces.items():
            if data.shared_docs:
                await self._writer.index(data.shared_docs, namespace)
                counts[namespace] = len(data.shared_docs)

        logger.info("[Indexer] Indexed shared docs: %s", counts)
        return counts


@dataclass(slots=True)
class _NamespaceData:
    """Internal storage for a registered namespace."""

    shared_docs: list[OrchidDocument]
    tenant_docs_fn: Callable[[str], list[OrchidDocument]] | None = None
