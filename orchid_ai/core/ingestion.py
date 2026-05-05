"""
Pluggable ingestion primitives.

Splits the parse → chunk → upsert pipeline at the chunking boundary so
new chunking approaches (semantic, hierarchical, headered, …) plug in
via a registry instead of editing :mod:`orchid_ai.documents.pipeline`.

Two ABCs:

  * :class:`OrchidIngestionStrategy` — turn parsed text into a list of
    :class:`OrchidChunk`.  Strategies own boundary choice, parent-child
    layout, and any LLM calls they need.
  * :class:`OrchidChunkPostProcessor` — augment a chunk list after
    splitting (e.g. prepend a contextual header, redact PII).  Composes
    with any strategy.

Both live in ``core/`` so the documents pipeline + per-tool override
can depend on them without risking a cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from .doc_store import OrchidDocStore
from .scopes import OrchidRAGScope


class OrchidChunk(NamedTuple):
    """One ingestion-pipeline chunk.

    ``text`` is what the embedder sees (and what gets BM25-encoded for
    hybrid search).  ``metadata`` carries everything else the writer
    needs: scope fields (``tenant_id`` / ``user_id`` / ``chat_id`` /
    ``scope``), provenance (``source_file`` / ``chunk_index``),
    parent linkage (``parent_id`` for hierarchical strategies), and any
    integrator-supplied filterable fields (``language`` / ``status`` /
    …) consumed by metadata-filter retrieval.
    """

    text: str
    metadata: dict[str, Any]


class OrchidIngestionStrategy(ABC):
    """Turn parsed document text into a list of :class:`OrchidChunk`.

    The signature is wide on purpose so strategies can opt into
    side-channel resources without forcing every caller to thread them
    through:

      * ``doc_store`` — for hierarchical strategies that store parent
        chunks under their own IDs.  Pass ``None`` (or a ``NullDocStore``)
        when irrelevant.
      * ``embeddings`` — for semantic strategies that detect topic
        boundaries with the same embedder used at retrieval time.

    Strategies must be **stateless** across calls — any per-document
    state lives inside ``ingest()``.  This makes them safe to register
    once at process startup and share across requests.
    """

    @abstractmethod
    async def ingest(
        self,
        *,
        text: str,
        filename: str,
        scope: OrchidRAGScope,
        doc_store: OrchidDocStore | None = None,
        embeddings: Any | None = None,
    ) -> list[OrchidChunk]:
        """Split ``text`` into chunks tagged with the right metadata.

        ``filename`` is purely informational (used for ``source_file``
        metadata) and may be empty for non-file ingestion paths.
        """
        ...


class OrchidChunkPostProcessor(ABC):
    """Augment a chunk list after splitting.

    Post-processors compose linearly: the YAML
    ``ingestion.post_processors: [contextual_headers, pii_redact]``
    runs each in order, the next consuming the prior's output.

    The signature mirrors :class:`OrchidIngestionStrategy.ingest` so
    processors that need the full original document (e.g. to derive a
    LLM-generated summary header) can refer back to it via the ``text``
    + ``filename`` kwargs without re-reading the file.

    ``graph_store`` / ``scope`` / ``schema`` are forwarded by the
    pipeline so post-processors that write side-effects (e.g.
    :class:`EntityExtractionPostProcessor` writing entities to the
    knowledge graph) can read them at call time.  Most processors
    ignore them via ``**_kwargs``.
    """

    @abstractmethod
    async def process(
        self,
        chunks: list[OrchidChunk],
        *,
        text: str,
        filename: str,
        chat_model: Any | None = None,
        graph_store: Any | None = None,
        scope: Any | None = None,
        schema: dict[str, Any] | None = None,
    ) -> list[OrchidChunk]:
        """Return a (possibly transformed) list of chunks.

        Implementations may shorten, lengthen, or reorder the list.
        ``chat_model`` / ``graph_store`` are duck-typed (``Any``) so
        this ABC stays free of the LangChain ``BaseChatModel`` and
        ``rag/backends/`` imports in ``core/``.  ``scope`` is the
        :class:`OrchidRAGScope` for the current ingestion call;
        ``schema`` is an optional per-namespace constraint dict (e.g.
        ``{"entity_types": ["supplier", "product"]}``).
        """
        ...
