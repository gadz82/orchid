"""RAG-related configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class OrchidRAGDefaultsConfig(BaseModel):
    """Default RAG settings inherited by all agents."""

    k: int = 5
    enabled: bool = True
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)
    reformulate_queries: bool = True  # rewrite queries using conversation history
    retriever_type: Literal["simple", "multi_query"] = "simple"
    #: Maximum number of characters of retrieved RAG context injected
    #: into the agent's system prompt.  The framework JSON-pretty-prints
    #: the retrieved docs and slices the result at this boundary.
    #: Increase for catalog-style agents whose RAG IS the source of
    #: truth (default 3000 truncates large catalogs mid-document).
    max_context_chars: int = 3000


class OrchidRAGConfig(BaseModel):
    """Per-agent RAG settings.

    ``retriever_type`` controls the retrieval strategy:
    - ``simple`` (default) — single vector similarity search per query.
    - ``multi_query`` — LLM generates query variations for broader
      recall, then results are merged and deduplicated.

    When ``retriever_type`` is ``None`` (the YAML default), the value
    is inherited from ``defaults.rag.retriever_type``.  Set it
    explicitly to override the default.

    ``max_context_chars`` (``None`` = inherit) sets the upper bound on
    how much of the retrieved RAG context is inlined into the agent's
    system prompt before truncation.  Bump it for agents whose RAG is
    authoritative (catalogs / reference data) so the LLM sees the full
    set instead of a slice.
    """

    namespace: str = ""
    k: int = 5
    enabled: bool = True
    reformulate_queries: bool = True  # rewrite queries using conversation history
    retriever_type: Literal["simple", "multi_query"] | None = None  # None = inherit from defaults
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)
    max_context_chars: int | None = None  # None = inherit from defaults.rag.max_context_chars
