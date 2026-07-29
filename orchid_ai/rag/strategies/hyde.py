"""
``HyDERetrieval`` — Hypothetical Document Embeddings retrieval.

The strategy generates ``n_hypothetical`` plausible answer paragraphs
via :class:`HyDETransformer`, then retrieves documents using both the
original query AND each hypothetical as the embedding target.
Results are merged by document id (highest score wins).

External ``pre_strategy=False`` transformers passed via the
``transformers`` kwarg compose with the internal HyDE step — pair
``strategy: hyde`` with ``query_transformers: [multi_query]`` for
HyDE×N at retrieval time.

Configuration: ``OrchidRetrievalConfig.hyde.n_hypothetical`` flows in
through :meth:`HyDERetrieval.from_config`.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from ...core.scopes import OrchidRAGScope
from ..transformers.hyde import HyDETransformer
from ._helpers import expand_queries, fan_out_retrieve

logger = logging.getLogger(__name__)


class HyDERetrieval(OrchidRetrievalStrategy):
    """Retrieve via the original query + ``n_hypothetical`` HyDE paragraphs."""

    def __init__(self, *, n_hypothetical: int = 1, retrieval_timeout: float = 30.0) -> None:
        self._n_hypothetical = n_hypothetical
        self._retrieval_timeout = retrieval_timeout

    @classmethod
    def from_config(cls, config: Any) -> HyDERetrieval:
        """Read ``config.hyde.n_hypothetical`` when available."""
        n = getattr(getattr(config, "hyde", None), "n_hypothetical", 1) if config else 1
        return cls(n_hypothetical=n)

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

        # Internal HyDE step — generate ``n_hypothetical`` plausible
        # answers and add them to the fan-out set.  No chat_model =>
        # we silently degrade to single-query retrieval (the strategy
        # is still useful as a no-op when the LLM is unavailable).
        if chat_model is not None:
            try:
                hypotheticals = await HyDETransformer(n_hypothetical=self._n_hypothetical).transform(
                    query, chat_model=chat_model
                )
                queries.extend(hypotheticals)
            except Exception as exc:
                logger.warning("[HyDERetrieval] Hypothetical generation failed: %s", exc)
        else:
            logger.debug("[HyDERetrieval] No chat_model — skipping hypothetical generation")

        # External transformer chain — composes orthogonally with the
        # internal HyDE step.  ``expand_queries`` skips pre_strategy
        # transformers (those already ran at agent entry).
        if transformers and chat_model is not None:
            queries.extend(await expand_queries(query, transformers, chat_model=chat_model))

        return await fan_out_retrieve(
            queries=queries,
            namespace=namespace,
            scope=scope,
            k=k,
            reader=reader,
            timeout=self._retrieval_timeout,
            metadata_filters=metadata_filters,
        )
