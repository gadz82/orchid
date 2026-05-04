"""RAG-related configuration models (ADR-022 / ADR-023 / ADR-027 / ADR-028).

The schema is split into three nested blocks:

  * :class:`OrchidIngestionConfig` — strategy + chunk-size knobs
    consumed by :mod:`orchid_ai.documents.strategies`.
  * :class:`OrchidRetrievalConfig` — strategy + transformer chain +
    metadata-filter dict consumed by :mod:`orchid_ai.rag.strategies`.
  * :class:`OrchidRAGConfig` — agent-level wrapper exposing both blocks
    plus the agent-scoped knobs (``namespace`` / ``k`` / TTL / context
    cap).

Per the ADRs the blocks land in both ``defaults.rag`` and
``agents.<name>.rag`` — the merger in :func:`schema_agent._apply_defaults`
inherits any unset fields from the defaults block.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema_prompts import OrchidQueryTransformerPromptsConfig


class OrchidIngestionConfig(BaseModel):
    """How documents are turned into chunks (ADR-022).

    ``strategy`` is a free-form string so integrators can register
    custom strategies via
    :func:`orchid_ai.documents.strategies.register_ingestion_strategy`.
    Stage 1 ships ``recursive``; Stage 2 adds ``semantic``,
    ``hierarchical``, ``headered``.

    ``post_processors`` runs in order after the strategy splits the
    text — Stage 2 introduces ``contextual_headers`` and Stage 5 adds
    ``entity_extraction`` for GraphRAG.

    ``parent_chunk_size > 0`` triggers the parent-in-metadata layout in
    ``RecursiveIngestion``; Stage 2's ``HierarchicalIngestion`` uses an
    :class:`OrchidDocStore` instead.

    ``None`` on ``strategy`` means "inherit from defaults".
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    parent_chunk_size: int = 0
    parent_chunk_overlap: int = 200
    post_processors: list[str] = Field(default_factory=list)


class OrchidHydeConfig(BaseModel):
    """Per-agent ``hyde`` retrieval knobs (ADR-023 §YAML).

    ``n_hypothetical`` controls how many hypothetical answers
    :class:`HyDERetrieval` (and :class:`HyDETransformer`, when
    constructed via ``from_config``) generate per query.  ``1`` is the
    classic HyDE — adding more grows recall at the cost of LLM calls.
    """

    model_config = ConfigDict(extra="forbid")

    n_hypothetical: int = 1


class OrchidHybridConfig(BaseModel):
    """Per-agent ``hybrid`` retrieval knobs (ADR-025 §YAML).

    ``sparse_encoder`` selects the :class:`OrchidSparseEncoder` from
    :mod:`orchid_ai.rag.sparse`'s registry — Stage 4 ships ``bm25`` as
    the default; ``splade`` lands behind the optional ``splade`` extra.

    ``fusion`` chooses between Reciprocal Rank Fusion (``rrf``,
    parameter-free, robust default) and weighted-linear fusion
    (``linear``, uses ``sparse_weight``).  ``rrf_k`` is the standard
    RRF constant (60 follows the Cormack et al. paper).
    """

    model_config = ConfigDict(extra="forbid")

    sparse_encoder: str = "bm25"
    sparse_weight: float = 0.4
    fusion: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = 60


class OrchidGraphRetrievalConfig(BaseModel):
    """Per-agent ``graph_rag`` retrieval knobs (ADR-026 §YAML).

    ``enabled`` is the ingestion-side feature flag — it lets the
    ``entity_extraction`` post-processor short-circuit when False so
    integrators using a non-graph strategy don't pay the LLM cost.

    ``max_hops`` caps the BFS depth from each seed entity.  ``2`` is a
    sensible default — most useful relations sit one or two hops away
    and deeper walks dilute the relevance signal.

    ``fuse_with_vectors`` controls whether the strategy returns the
    serialised sub-graph alongside vector hits (``True``, default) or
    on its own (``False``, "graph context only" mode for tests / debug).

    ``relation_filter`` narrows traversal to specific edge labels —
    e.g. ``[reports_to, manages]`` for an org-chart agent.  Empty list
    means "all relations" per ADR-026.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_hops: int = 2
    fuse_with_vectors: bool = True
    relation_filter: list[str] = Field(default_factory=list)


class OrchidRetrievalConfig(BaseModel):
    """How a query is turned into ranked results (ADR-023 / ADR-027).

    ``strategy`` selects the :class:`OrchidRetrievalStrategy` from the
    registry — Stage 1 ships ``simple`` and ``multi_query``; Stage 3
    adds ``hyde``; later stages add ``hybrid``, ``graph_rag``.

    ``query_transformers`` is an ordered list of transformer names.
    The agent applies ``pre_strategy=True`` transformers (e.g.
    ``reformulate``) at turn entry so the rewritten query feeds RAG and
    the agentic loop alike; ``pre_strategy=False`` transformers (e.g.
    ``multi_query``, ``hyde``, ``decompose``) are forwarded to the
    strategy and fanned out internally.

    ``metadata_filters`` is the operator mini-language defined in
    ADR-027 (used from Stage 6 onward).

    Per-strategy blocks (``hyde``, future ``hybrid``, ``graph``) live
    here so a single :class:`OrchidRetrievalConfig` carries every knob
    a strategy reads through its ``from_config`` classmethod.

    ``None`` on ``strategy`` / ``query_transformers`` means
    "inherit from defaults".
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str | None = None
    query_transformers: list[str] | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    hyde: OrchidHydeConfig = Field(default_factory=OrchidHydeConfig)
    hybrid: OrchidHybridConfig = Field(default_factory=OrchidHybridConfig)
    graph: OrchidGraphRetrievalConfig = Field(default_factory=OrchidGraphRetrievalConfig)

    #: Optional prompt overrides for the four built-in query transformers.
    #: ``None`` on any field within means "use the module-level default
    #: shipped with Orchid".  See
    #: :class:`OrchidQueryTransformerPromptsConfig` for the field list.
    transformer_prompts: OrchidQueryTransformerPromptsConfig = Field(
        default_factory=OrchidQueryTransformerPromptsConfig,
    )


class OrchidRAGDefaultsConfig(BaseModel):
    """Default RAG settings inherited by all agents."""

    model_config = ConfigDict(extra="forbid")

    k: int = 5
    enabled: bool = True
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)
    #: Maximum number of characters of retrieved RAG context injected
    #: into the agent's system prompt.  Increase for catalog-style
    #: agents whose RAG IS the source of truth (default 3000 truncates
    #: large catalogs mid-document).
    max_context_chars: int = 3000
    ingestion: OrchidIngestionConfig = Field(default_factory=OrchidIngestionConfig)
    retrieval: OrchidRetrievalConfig = Field(default_factory=OrchidRetrievalConfig)


class OrchidRAGConfig(BaseModel):
    """Per-agent RAG settings.

    Agent-level fields override matching ``defaults.rag.*`` fields when
    set; ``None`` means "inherit from defaults".  ``ingestion`` and
    ``retrieval`` blocks are merged shallowly (each block's fields
    inherit independently — see :func:`schema_agent._apply_defaults`).
    """

    model_config = ConfigDict(extra="forbid")

    namespace: str = ""
    k: int = 5
    enabled: bool = True
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)
    max_context_chars: int | None = None  # None = inherit from defaults.rag.max_context_chars
    ingestion: OrchidIngestionConfig = Field(default_factory=OrchidIngestionConfig)
    retrieval: OrchidRetrievalConfig = Field(default_factory=OrchidRetrievalConfig)
