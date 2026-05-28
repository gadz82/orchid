"""
LangChain-compatible retriever wrapper for Orchid's vector store.

``OrchidRetriever`` wraps the framework's ``OrchidVectorReader`` ABC as a
LangChain ``BaseRetriever``, enabling composition with any LangChain
retriever chain.

Multi-query retrieval no longer lives here — it landed in
:class:`orchid_ai.rag.strategies.MultiQueryRetrieval` (which delegates
the variation-generation prompt to
:class:`orchid_ai.rag.transformers.MultiQueryTransformer`).  Use the
strategy registry rather than calling a free function::

    from orchid_ai.rag.strategies import get_retrieval_strategy

    strategy = get_retrieval_strategy("multi_query")
    results = await strategy.retrieve(
        query=q, namespace="kb", scope=scope, k=5,
        reader=reader, chat_model=chat_model,
    )
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.repository import OrchidSearchResult
from .adapters import to_langchain_document


class OrchidRetriever(BaseRetriever):
    """Wraps Orchid's ``OrchidVectorReader`` as a LangChain ``BaseRetriever``.

    All fields use ``Any`` type hints to satisfy Pydantic validation
    without importing concrete types that would violate dependency rules.

    Conversion from the framework's :class:`OrchidDocument` to
    LangChain's ``Document`` happens here at the boundary via
    :func:`orchid_ai.rag.adapters.to_langchain_document` — keeping
    ``core/`` free of LangChain imports.
    """

    reader: Any  # OrchidVectorReader — Any to avoid Pydantic ABC validation issues
    namespace: str
    scope: Any  # OrchidRAGScope
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
        results: list[OrchidSearchResult] = await self.reader.retrieve(
            query=query,
            namespace=self.namespace,
            k=self.k,
            scope=self.scope,
        )
        return [to_langchain_document(r.document) for r in results]
