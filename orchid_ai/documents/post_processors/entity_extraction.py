"""
Entity extraction during ingestion.

Two collaborating pieces land here:

* :class:`LLMEntityExtractor` — the reference :class:`OrchidEntityExtractor`.
  Calls ``chat_model.with_structured_output(_Extraction)`` to coerce the
  LLM's response into a Pydantic schema, then converts the validated
  result into :class:`OrchidEntity` / :class:`OrchidEdge` instances.
* :class:`EntityExtractionPostProcessor` — the
  :class:`OrchidChunkPostProcessor` that runs per chunk during
  ingestion, writes the extracted entities + edges to an
  :class:`OrchidGraphStore`, and tags every chunk with a
  ``mentioned_entities`` metadata field so retrieval-time filtering
  can target chunks that mention specific entities.

The post-processor is **opt-in**: when no ``chat_model`` or
``graph_store`` is supplied, ``process()`` returns the chunks unchanged
(no extraction, no writes).  This means an integrator who registers
``entity_extraction`` in YAML but doesn't wire a graph store gets a
graceful degradation rather than a crash.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ...core.graph_store import (
    OrchidEdge,
    OrchidEntity,
    OrchidEntityExtractor,
    OrchidGraphStore,
)
from ...core.ingestion import OrchidChunk, OrchidChunkPostProcessor
from ...core.scopes import OrchidRAGScope

logger = logging.getLogger(__name__)


# ── Structured-output Pydantic schema ─────────────────────────


class _ExtractedEntity(BaseModel):
    """Pydantic shape used with ``chat_model.with_structured_output``."""

    id: str = Field(description="Stable identifier, e.g. 'supplier:acme' or 'person:jane_doe'.")
    type: str = Field(description="Entity type, e.g. 'supplier', 'product', 'person'.")
    name: str = Field(description="Human-readable display name for the entity.")
    properties: dict[str, Any] = Field(default_factory=dict)


class _ExtractedEdge(BaseModel):
    source_id: str = Field(description="Source entity id (must match an extracted entity).")
    target_id: str = Field(description="Target entity id (must match an extracted entity).")
    relation: str = Field(description="Relationship label, e.g. 'supplies', 'reports_to'.")
    properties: dict[str, Any] = Field(default_factory=dict)


class OrchidExtractedGraph(BaseModel):
    """Top-level structured-output schema returned by the LLM.

    Public so integrators wiring custom extractors can subclass and
    extend the schema without re-implementing the prompt boilerplate.
    """

    entities: list[_ExtractedEntity] = Field(default_factory=list)
    edges: list[_ExtractedEdge] = Field(default_factory=list)


_DEFAULT_EXTRACTION_PROMPT = (
    "You extract structured entities and relationships from text.\n"
    "RULES:\n"
    "- Use stable lowercase ids prefixed with the entity type "
    "(e.g. 'supplier:acme', 'person:jane_doe').\n"
    "- Every edge's source_id and target_id MUST appear in the entities list.\n"
    "- Stick to the canonical relation labels in the schema when one fits;"
    " invent only when none of the canonical labels apply.\n"
    "- Do NOT invent entities not grounded in the input text."
)


class LLMEntityExtractor(OrchidEntityExtractor):
    """LLM-driven entity + edge extractor reference implementation.

    Falls back to ``([], [])`` on any LLM error so a flaky model
    doesn't sink the entire ingestion run — extraction is best-effort
    and the writer side can always retry the same chunk later.
    """

    def __init__(self, *, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or _DEFAULT_EXTRACTION_PROMPT

    async def extract(
        self,
        text: str,
        *,
        chat_model: Any,
        schema: dict[str, Any] | None = None,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        if chat_model is None or not text.strip():
            return ([], [])

        prompt = self._system_prompt
        if schema:
            constraints = []
            entity_types = schema.get("entity_types") or []
            relations = schema.get("relations") or []
            if entity_types:
                constraints.append(f"Allowed entity types: {', '.join(entity_types)}.")
            if relations:
                constraints.append(f"Allowed relations: {', '.join(relations)}.")
            if constraints:
                prompt = prompt + "\n\nADDITIONAL CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints)

        try:
            structured = chat_model.with_structured_output(OrchidExtractedGraph)
            result = await structured.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=text),
                ]
            )
        except Exception as exc:
            logger.warning("[LLMEntityExtractor] extraction failed: %s", exc)
            return ([], [])

        return self._validate(result)

    @staticmethod
    def _validate(
        result: OrchidExtractedGraph,
    ) -> tuple[list[OrchidEntity], list[OrchidEdge]]:
        """Drop edges that reference unknown entity ids — the LLM
        occasionally hallucinates dangling endpoints."""
        entity_ids = {e.id for e in result.entities}
        entities = [
            OrchidEntity(
                id=e.id,
                type=e.type,
                name=e.name,
                properties=dict(e.properties),
                metadata={},
            )
            for e in result.entities
        ]
        edges: list[OrchidEdge] = []
        for e in result.edges:
            if e.source_id not in entity_ids or e.target_id not in entity_ids:
                logger.debug(
                    "[LLMEntityExtractor] dropping dangling edge %s -[%s]-> %s",
                    e.source_id,
                    e.relation,
                    e.target_id,
                )
                continue
            edges.append(
                OrchidEdge(
                    source_id=e.source_id,
                    target_id=e.target_id,
                    relation=e.relation,
                    properties=dict(e.properties),
                    metadata={},
                )
            )
        return (entities, edges)


# ── Chunk post-processor ─────────────────────────────────────


class EntityExtractionPostProcessor(OrchidChunkPostProcessor):
    """Extract entities + edges from each chunk and write them to the graph store.

    The constructor is parameterless so the registry default
    (``register_post_processor("entity_extraction", EntityExtractionPostProcessor)``)
    works out of the box; the per-call dependencies (``chat_model``,
    ``graph_store``, ``scope``, ``schema``) flow through the
    :meth:`OrchidChunkPostProcessor.process` kwargs the pipeline forwards.

    Custom extractors are passed via the ``extractor`` constructor
    kwarg — integrators using a rule-based or domain-specific
    extractor wire it once at composition time.
    """

    def __init__(self, *, extractor: OrchidEntityExtractor | None = None) -> None:
        self._extractor = extractor or LLMEntityExtractor()

    async def process(
        self,
        chunks: list[OrchidChunk],
        *,
        text: str,
        filename: str,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        scope: OrchidRAGScope | None = None,
        schema: dict[str, Any] | None = None,
    ) -> list[OrchidChunk]:
        # Graceful no-op when dependencies are missing — keeps the
        # post-processor harmless when wired in YAML but the integrator
        # hasn't injected a graph store yet.
        if not chunks:
            return []
        if chat_model is None or graph_store is None or scope is None:
            logger.debug(
                "[EntityExtractionPostProcessor] missing dependency "
                "(chat_model=%s, graph_store=%s, scope=%s) — skipping",
                chat_model is not None,
                graph_store is not None,
                scope is not None,
            )
            return chunks

        # When the graph store is the no-op fallback (NullGraphStore),
        # extraction would burn LLM cost for nothing — short-circuit.
        if getattr(graph_store, "is_null", False):
            return chunks

        out: list[OrchidChunk] = []
        for chunk in chunks:
            entities, edges = await self._extractor.extract(
                chunk.text,
                chat_model=chat_model,
                schema=schema,
            )
            if entities:
                await graph_store.upsert_entities(entities, scope)
            if edges:
                await graph_store.upsert_edges(edges, scope)
            mentioned = sorted({e.id for e in entities})
            out.append(
                OrchidChunk(
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "mentioned_entities": mentioned,
                    },
                )
            )
        return out
