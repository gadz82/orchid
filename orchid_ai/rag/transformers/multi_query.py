"""
``MultiQueryTransformer`` — generate N alternative queries via LLM.

Marked ``pre_strategy=False``: the strategy that consumes this
transformer fans out the returned queries (parallel retrievals + score
merge), and the original query is **not** included in the returned
list — strategies typically prepend it themselves.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.retrieval import OrchidQueryTransformer

logger = logging.getLogger(__name__)


DEFAULT_MULTI_QUERY_PROMPT = (
    "You are a search query generator.  Given a user question, generate "
    "{n} alternative search queries that would help retrieve relevant "
    "documents.  The queries should cover different phrasings, synonyms, "
    "and aspects of the original question.\n"
    "Output ONLY the queries, one per line.  No numbering, no explanation."
)


class MultiQueryTransformer(OrchidQueryTransformer):
    """Ask an LLM for ``num_queries`` alternative phrasings of the query.

    The system prompt template is configurable via the ``system_prompt``
    constructor kwarg (or YAML
    ``rag.retrieval.transformer_prompts.multi_query``).  ``None`` keeps
    :data:`DEFAULT_MULTI_QUERY_PROMPT`.  Custom templates must keep the
    ``{n}`` placeholder so the runtime can substitute the variation
    count.
    """

    pre_strategy: ClassVar[bool] = False

    def __init__(
        self,
        *,
        num_queries: int = 3,
        timeout_seconds: float = 15.0,
        system_prompt: str | None = None,
    ) -> None:
        self._num_queries = num_queries
        self._timeout_seconds = timeout_seconds
        self._system_prompt = system_prompt or DEFAULT_MULTI_QUERY_PROMPT

    async def transform(
        self,
        query: str,
        *,
        chat_model: Any,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        if chat_model is None:
            return []

        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await chat_model.ainvoke(
                    [
                        SystemMessage(content=self._system_prompt.format(n=self._num_queries)),
                        HumanMessage(content=query),
                    ]
                )
            lines = [line.strip() for line in (result.content or "").split("\n") if line.strip()]
            return lines[: self._num_queries]
        except Exception as exc:
            # Broad catch on purpose — variation generation is best-effort
            # and the caller (the strategy) gracefully handles an empty
            # variation list by falling back to the single original query.
            logger.warning("[MultiQueryTransformer] Failed to generate variations: %s", exc)
            return []
