"""
``MultiQueryRetrieval`` — fan out the query into N variations, merge.

Always runs an internal :class:`MultiQueryTransformer` to generate
``num_queries`` LLM-driven variations.  Composes orthogonally with
external ``pre_strategy=False`` transformers passed via the
``transformers`` kwarg — pair ``strategy: multi_query`` with
``query_transformers: [hyde, decompose]`` to fan out via every
declared transformer in addition to the internal multi-query step.

Without a ``chat_model``, the strategy degrades to single-query
retrieval (the variation step is skipped).
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from ...core.scopes import OrchidRAGScope
from ..transformers.multi_query import MultiQueryTransformer
from ._helpers import expand_queries, fan_out_retrieve

logger = logging.getLogger(__name__)


class MultiQueryRetrieval(OrchidRetrievalStrategy):
    """Generate N variations, retrieve in parallel, merge by score."""

    def __init__(self, *, num_queries: int = 3, retrieval_timeout: float = 30.0) -> None:
        self._num_queries = num_queries
        self._retrieval_timeout = retrieval_timeout

    async def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        scope: OrchidRAGScope,
        k: int,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        doc_store: OrchidDocStore | None = None,
        transformers: list[OrchidQueryTransformer] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        queries: list[str] = [query]

        # Internal multi-query step — generate ``num_queries`` LLM
        # variations.  No chat_model => single-query retrieval.
        if chat_model is not None:
            try:
                variations = await MultiQueryTransformer(num_queries=self._num_queries).transform(
                    query, chat_model=chat_model
                )
                queries.extend(variations)
            except Exception as exc:
                logger.warning("[MultiQueryRetrieval] Variation generation failed: %s", exc)
        else:
            logger.debug("[MultiQueryRetrieval] No chat_model — falling back to single-query retrieval")

        # External transformer chain — composes with the internal step.
        if transformers and chat_model is not None:
            queries.extend(await expand_queries(query, transformers, chat_model=chat_model))

        return await fan_out_retrieve(
            queries=queries,
            namespace=namespace,
            scope=scope,
            k=k,
            reader=reader,
            timeout=self._retrieval_timeout,
        )
