"""
``DecomposeTransformer`` — split a complex query into independent sub-queries.

Multi-part questions ("Compare X with Y in terms of Z") often retrieve
better when each sub-aspect is searched separately and the results
merged.  The transformer asks the LLM to extract independent
sub-questions; the consuming strategy fans them out and merges.

Marked ``pre_strategy=False``: it produces N sub-queries that the
strategy fans out — different from ``reformulate``, which produces
exactly one replacement query at agent entry.

For simple, already-atomic queries the LLM is instructed to return
the original on a single line — the strategy then runs a normal
fan-out of `[original, original]` which dedupes to one retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.retrieval import OrchidQueryTransformer

logger = logging.getLogger(__name__)


DEFAULT_DECOMPOSE_PROMPT = (
    "You decompose complex multi-part questions into independent sub-questions, "
    "each of which can be answered without reference to the others.\n"
    "RULES:\n"
    "- Output AT MOST {n} sub-questions, one per line.\n"
    "- If the question is already atomic, output it on a single line unchanged.\n"
    "- Each sub-question must stand on its own — no pronouns referring back to "
    "  other sub-questions.\n"
    "- No numbering, no preamble, no explanation."
)


class DecomposeTransformer(OrchidQueryTransformer):
    """Split a query into ``<= max_sub_queries`` independent sub-queries.

    The system prompt is configurable via the ``system_prompt``
    constructor kwarg (or YAML
    ``rag.retrieval.transformer_prompts.decompose``).  Custom templates
    must keep the ``{n}`` placeholder so the runtime can substitute
    the maximum sub-query count.
    """

    pre_strategy: ClassVar[bool] = False

    def __init__(
        self,
        *,
        max_sub_queries: int = 4,
        timeout_seconds: float = 20.0,
        system_prompt: str | None = None,
    ) -> None:
        if max_sub_queries < 2:
            raise ValueError(f"max_sub_queries must be >= 2; got {max_sub_queries}")
        self._max_sub_queries = max_sub_queries
        self._timeout_seconds = timeout_seconds
        self._system_prompt = system_prompt or DEFAULT_DECOMPOSE_PROMPT

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
                        SystemMessage(content=self._system_prompt.format(n=self._max_sub_queries)),
                        HumanMessage(content=query),
                    ],
                    temperature=0.0,
                )
        except Exception as exc:
            # Best-effort — the strategy keeps the original query.
            logger.warning("[DecomposeTransformer] Failed: %s", exc)
            return []

        lines = [line.strip() for line in (result.content or "").split("\n") if line.strip()]
        return lines[: self._max_sub_queries]
