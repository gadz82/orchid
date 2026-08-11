"""Ingestion manifest — track indexed source files for idempotent re-runs.

This module lives in ``core/`` and therefore depends only on the Python
standard library.  Concrete manifest stores (SQLite, PostgreSQL) live in
``persistence/`` and ``orchid-storage-postgres/`` respectively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OrchidIngestionManifest(ABC):
    """Track which source files have been indexed into which namespace.

    The manifest stores a content hash per ``(source_id, namespace)`` so
    callers can skip unchanged files across indexer runs.  It also records
    the vector document IDs produced during ingestion so that stale vectors
    can be deleted when a source is removed or re-indexed.

    Implementations are responsible for their own connection lifecycle;
    callers should ``await manifest.close()`` when done.
    """

    @abstractmethod
    async def should_skip(self, source_id: str, content_hash: str, namespace: str) -> bool:
        """Return ``True`` if the source has already been indexed with the same hash.

        Parameters
        ----------
        source_id:
            Stable identifier for the source file or document (e.g. a relative
            path or a front-matter ``page_id``).
        content_hash:
            Hash of the raw source content (e.g. SHA-256 hex digest).
        namespace:
            Target vector-store namespace / collection.
        """
        ...

    @abstractmethod
    async def record(
        self,
        source_id: str,
        content_hash: str,
        namespace: str,
        document_ids: list[str],
    ) -> None:
        """Record or update the manifest entry for an indexed source.

        This is an upsert: if a row already exists for ``(source_id,
        namespace)`` it is replaced with the new hash and document IDs.
        """
        ...

    @abstractmethod
    async def remove(self, source_id: str, namespace: str) -> None:
        """Remove the manifest entry for a source."""
        ...

    @abstractmethod
    async def list_known(self, namespace: str) -> set[str]:
        """Return all source IDs currently recorded for a namespace."""
        ...

    @abstractmethod
    async def get_document_ids(self, source_id: str, namespace: str) -> list[str]:
        """Return the vector document IDs recorded for a source.

        Returns an empty list if the source is not known.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...
