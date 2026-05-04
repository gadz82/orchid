"""
``SimpleRetrieval`` — single-query, single-namespace dense retrieval.

The default strategy.  Wraps :meth:`OrchidVectorReader.retrieve` with no
extra logic — same shape as the pre-redesign hot path.  Strategies
that fan out (``multi_query``, ``hyde``, future ``hybrid`` /
``graph_rag``) replace this one on a per-agent basis via the
``retrieval.strategy`` YAML field.

By design ``SimpleRetrieval`` **ignores** non-``pre_strategy``
transformers passed via ``transformers``.  Users who want fan-out
should pick a fan-out strategy.  When a non-empty ``transformers``
list is supplied we emit a one-line debug log to make the mismatch
discoverable.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.doc_store import OrchidDocStore
from ...core.graph_store import OrchidGraphStore
from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer, OrchidRetrievalStrategy
from ...core.scopes import OrchidRAGScope

logger = logging.getLogger(__name__)


class SimpleRetrieval(OrchidRetrievalStrategy):
    """Single dense ``reader.retrieve`` call.

    Optional ``transformers`` / ``graph_store`` / ``doc_store`` /
    ``metadata_filters`` are accepted for ABC parity but ignored —
    fan-out / hydration / filtering belong to other strategies.
    """

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
        non_pre = [t for t in (transformers or []) if not t.pre_strategy]
        if non_pre:
            logger.debug(
                "[SimpleRetrieval] %d non-pre_strategy transformer(s) supplied but ignored — "
                "use strategy: multi_query or hyde for fan-out behaviour.",
                len(non_pre),
            )
        return await reader.retrieve(
            query=query,
            namespace=namespace,
            k=k,
            scope=scope,
            metadata_filters=metadata_filters,
        )
