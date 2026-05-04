"""
``InMemoryDocStore`` — dict-backed :class:`OrchidDocStore` (ADR-022 / ADR-028).

Suitable for tests, single-process demos, and integrators who don't
need cross-process persistence.  Idempotent ``put`` (the same
``doc_id`` always replaces).  ``get_many`` is a single dict lookup.
"""

from __future__ import annotations

from typing import Any

from ...core.doc_store import OrchidDocStore


class InMemoryDocStore(OrchidDocStore):
    """A simple in-process doc store backed by a dict.

    Not thread-safe across event loops.  Intended for single-process
    workloads (CLI runs, test fixtures, demos).
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, dict[str, Any]]] = {}

    async def put(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        self._store[doc_id] = (content, dict(metadata))

    async def get(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        record = self._store.get(doc_id)
        if record is None:
            return None
        content, metadata = record
        return content, dict(metadata)

    async def get_many(self, doc_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        out: dict[str, tuple[str, dict[str, Any]]] = {}
        for doc_id in doc_ids:
            record = self._store.get(doc_id)
            if record is not None:
                content, metadata = record
                out[doc_id] = (content, dict(metadata))
        return out
