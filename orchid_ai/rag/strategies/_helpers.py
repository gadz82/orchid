"""
Shared fan-out helpers for retrieval strategies.

Two utilities used by ``MultiQueryRetrieval``, ``HyDERetrieval``, and
future strategies that fan out the query into N variations:

  * :func:`expand_queries` — walk a list of ``pre_strategy=False``
    transformers and collect their outputs.  Per-transformer failures
    are logged and dropped so one broken transformer can't sink the
    whole retrieval.
  * :func:`fan_out_retrieve` — issue one ``reader.retrieve`` per query
    in parallel, dedupe results by document id (keeping the highest
    score), and return the top-``k``.  Wraps the parallel block in a
    timeout and falls back to single-query retrieval on expiry.

Strategies stay small and uniform — each one's ``retrieve()`` becomes
"build query set, hand to fan_out_retrieve".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...core.repository import OrchidSearchResult, OrchidVectorReader
from ...core.retrieval import OrchidQueryTransformer
from ...core.scopes import OrchidRAGScope

logger = logging.getLogger(__name__)


async def expand_queries(
    query: str,
    transformers: list[OrchidQueryTransformer],
    *,
    chat_model: Any,
) -> list[str]:
    """Run each ``pre_strategy=False`` transformer and collect new queries.

    The original ``query`` is **not** included — callers prepend it to
    the returned list (the original is always retrieved alongside any
    fan-out).  ``pre_strategy=True`` transformers in the list are
    silently skipped — they belong to ``apply_pre_strategy`` (the
    agent-entry path) and would otherwise duplicate work.

    Per-transformer exceptions are logged and dropped so a single
    misbehaving transformer (e.g. a flaky LLM call inside HyDE) never
    fails the entire retrieval.
    """
    expanded: list[str] = []
    for transformer in transformers:
        if transformer.pre_strategy:
            continue
        try:
            new_queries = await transformer.transform(query, chat_model=chat_model)
        except Exception as exc:
            logger.warning("[expand_queries] %s failed: %s", type(transformer).__name__, exc)
            continue
        expanded.extend(new_queries)
    return expanded


async def fan_out_retrieve(
    *,
    queries: list[str],
    namespace: str,
    scope: OrchidRAGScope,
    k: int,
    reader: OrchidVectorReader,
    timeout: float = 30.0,
    metadata_filters: dict[str, Any] | None = None,
) -> list[OrchidSearchResult]:
    """Retrieve in parallel for every query, dedupe by id (highest score wins).

    Returns at most ``k`` results, sorted by descending score.  On
    timeout, falls back to a single retrieval with the first query and
    returns whatever it finds (or ``[]`` on a second failure).
    ``metadata_filters`` flow through to every parallel
    ``reader.retrieve`` call so the strategy's filter declaration is
    consistently applied across the fan-out.
    """
    if not queries:
        return []

    tasks = [
        reader.retrieve(
            query=q,
            namespace=namespace,
            k=k,
            scope=scope,
            metadata_filters=metadata_filters,
        )
        for q in queries
    ]
    try:
        async with asyncio.timeout(timeout):
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
    except TimeoutError:
        logger.warning(
            "[fan_out_retrieve] Timed out after %ss with %d queries — single-query fallback",
            timeout,
            len(queries),
        )
        try:
            all_results = [
                await reader.retrieve(
                    query=queries[0],
                    namespace=namespace,
                    k=k,
                    scope=scope,
                    metadata_filters=metadata_filters,
                )
            ]
        except Exception as exc:
            logger.warning("[fan_out_retrieve] Fallback retrieve failed: %s", exc)
            return []

    seen: dict[str, OrchidSearchResult] = {}
    for rs in all_results:
        if isinstance(rs, BaseException):
            logger.warning("[fan_out_retrieve] One query failed: %s", rs)
            continue
        for sr in rs:
            doc_id = sr.document.id or sr.document.page_content[:100]
            existing = seen.get(doc_id)
            if existing is None or sr.score > existing.score:
                seen[doc_id] = sr

    merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
    return merged[:k]
