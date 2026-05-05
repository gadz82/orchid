"""
Parent-document storage primitive.

Hierarchical RAG retrieves child chunks for precision then hydrates the
larger parent context for the LLM.  ``OrchidDocStore`` is the put / get
surface used by the hierarchical ingestion strategy and the matching
retrieval-time hydration step — kept independent of the vector backend
so an integrator can pair (e.g.) Qdrant for vectors with an in-memory
or relational doc store.

Lives in ``core/`` and depends only on stdlib types so every other
package — backends, strategies, runtime — can depend on it without
risking a cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class OrchidDocStore(ABC):
    """Persist arbitrary text blobs by stable ID.

    The contract is intentionally narrow: ``put`` writes one document,
    ``get`` reads one, ``get_many`` is the batched variant retrieval
    strategies use to hydrate a list of parent IDs in a single round
    trip.  Implementations may add their own batched ``put`` overloads
    on top — the ABC stays minimal.

    ``is_null`` is a class-level marker so strategies can detect the
    no-op fallback (``NullDocStore``) without crossing the
    ``documents/`` → ``rag/`` dependency line.  Concrete impls leave
    it ``False``; ``NullDocStore`` overrides to ``True`` so
    hierarchical ingestion can fall back to parent-in-metadata mode.
    """

    is_null: ClassVar[bool] = False

    @abstractmethod
    async def put(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Store a document under ``doc_id``.

        Implementations must be idempotent: writing the same ``doc_id``
        twice replaces the prior content + metadata.
        """
        ...

    @abstractmethod
    async def get(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        """Return ``(content, metadata)`` for ``doc_id`` or ``None`` if missing."""
        ...

    @abstractmethod
    async def get_many(self, doc_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        """Batched fetch.

        Missing IDs are silently dropped — the caller diffs the requested
        list against the returned keys to detect them.
        """
        ...
