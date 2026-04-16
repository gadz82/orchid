"""
Advanced retriever wrappers for Orchid's RAG pipeline.

``OrchidRetriever`` wraps the framework's ``VectorReader`` ABC as a
LangChain ``BaseRetriever``, enabling composition with any LangChain
retriever chain.

``multi_query_retrieve()`` implements the multi-query pattern: the LLM
generates query variations, each is run through the reader, and results
are deduplicated and merged by score.  This improves recall for vague
or multi-faceted queries without requiring the ``langchain`` package.

Example (standalone)::

    retriever = OrchidRetriever(
        reader=qdrant_reader,
        namespace="learning",
        scope=scope,
        k=5,
    )
    docs = await retriever.ainvoke("What is machine learning?")

Example (multi-query via GenericAgent)::

    # In agents.yaml:
    agents:
      learning:
        rag:
          retriever_type: multi_query
          k: 5
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever

from ..core.repository import SearchResult, VectorReader
from ..core.scopes import RAGScope

logger = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT = (
    "You are a search query generator.  Given a user question, generate "
    "{n} alternative search queries that would help retrieve relevant "
    "documents.  The queries should cover different phrasings, synonyms, "
    "and aspects of the original question.\n"
    "Output ONLY the queries, one per line.  No numbering, no explanation."
)


class OrchidRetriever(BaseRetriever):
    """Wraps Orchid's ``VectorReader`` as a LangChain ``BaseRetriever``.

    This enables composition with LangChain retriever chains and
    provides a standard ``ainvoke()`` / ``invoke()`` interface.

    All fields use ``Any`` type hints to satisfy Pydantic validation
    without importing concrete types that would violate dependency rules.
    """

    reader: Any  # VectorReader — Any to avoid Pydantic ABC validation issues
    namespace: str
    scope: Any  # RAGScope
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raise NotImplementedError("OrchidRetriever is async-only; use ainvoke()")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        results: list[SearchResult] = await self.reader.retrieve(
            query=query,
            namespace=self.namespace,
            k=self.k,
            scope=self.scope,
        )
        return [r.document for r in results]


async def _generate_query_variations(
    chat_model: Any,
    query: str,
    n: int = 3,
) -> list[str]:
    """Use the LLM to generate *n* alternative search queries."""
    try:
        result = await chat_model.ainvoke(
            [
                SystemMessage(content=_MULTI_QUERY_PROMPT.format(n=n)),
                HumanMessage(content=query),
            ]
        )
        lines = [line.strip() for line in (result.content or "").split("\n") if line.strip()]
        return lines[:n]
    except Exception as exc:
        logger.warning("[MultiQuery] Failed to generate variations: %s", exc)
        return []


async def multi_query_retrieve(
    query: str,
    reader: VectorReader,
    namespace: str,
    scope: RAGScope,
    chat_model: Any,
    k: int = 5,
    num_queries: int = 3,
) -> list[SearchResult]:
    """Multi-query retrieval: generate query variations, retrieve for each, deduplicate.

    The original query is always included.  Results are merged, deduplicated
    by document ID, and sorted by score (highest first).  Returns at most
    *k* results.

    Parameters
    ----------
    query : str
        Original user query.
    reader : VectorReader
        Vector store backend.
    namespace : str
        Collection name.
    scope : RAGScope
        Hierarchical scope filter.
    chat_model
        LangChain ``BaseChatModel`` for generating query variations.
    k : int
        Maximum results to return.
    num_queries : int
        Number of alternative queries to generate (default 3).

    Returns
    -------
    list[SearchResult]
        Deduplicated, score-sorted results (at most *k*).
    """
    # Generate alternative queries
    variations = await _generate_query_variations(chat_model, query, n=num_queries)
    all_queries = [query] + variations

    logger.info(
        "[MultiQuery] Retrieving with %d queries (original + %d variations)",
        len(all_queries),
        len(variations),
    )

    # Retrieve in parallel for all queries
    tasks = [reader.retrieve(query=q, namespace=namespace, k=k, scope=scope) for q in all_queries]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge and deduplicate by document ID, keeping highest score
    seen: dict[str, SearchResult] = {}
    for result_set in all_results:
        if isinstance(result_set, Exception):
            logger.warning("[MultiQuery] One query variation failed: %s", result_set)
            continue
        for sr in result_set:
            doc_id = sr.document.id or sr.document.page_content[:100]
            existing = seen.get(doc_id)
            if existing is None or sr.score > existing.score:
                seen[doc_id] = sr

    # Sort by score and return top-k
    merged = sorted(seen.values(), key=lambda sr: sr.score, reverse=True)
    return merged[:k]
