"""
``HyDETransformer`` — generate hypothetical document(s) that answer the query.

The HyDE pattern (Hypothetical Document Embeddings) embeds a
plausible answer to the query and retrieves documents similar to it.
The hypothetical does not need to be factually correct — it just
needs to look like a passage that *would* answer the question, so the
embedding lands in the right neighbourhood.

Marked ``pre_strategy=False``: the strategy that consumes this
transformer fans out the returned hypothetical(s) (one retrieve per
hypothetical) plus the original query, then merges results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.retrieval import OrchidQueryTransformer

logger = logging.getLogger(__name__)


DEFAULT_SINGLE_PROMPT = (
    "You are a hypothetical-document generator.  Given a user question, write a "
    "single concise paragraph (3-5 sentences) that, if it appeared in a document, "
    "would directly answer the question.  Use a confident, encyclopedic tone — the "
    "paragraph does NOT need to be factually correct, only plausibly written.  "
    "Output ONLY the paragraph; no preamble, no explanation."
)

DEFAULT_MULTI_PROMPT = (
    "You are a hypothetical-document generator.  Given a user question, write "
    "{n} distinct concise paragraphs (3-5 sentences each) that each, if they "
    "appeared in a document, would directly answer the question.  Vary phrasing "
    "and angle so each paragraph covers a different facet of the answer.  Use "
    "a confident, encyclopedic tone — the paragraphs do NOT need to be factually "
    "correct, only plausibly written.  Output one paragraph per line.  Do not "
    "number the paragraphs."
)


class HyDETransformer(OrchidQueryTransformer):
    """Ask an LLM for ``n_hypothetical`` plausible answers to the query.

    Both the single-paragraph and multi-paragraph prompts are
    customisable via constructor kwargs (or YAML
    ``rag.retrieval.transformer_prompts.hyde.{single,multi}``).
    Custom multi-paragraph templates must keep the ``{n}`` placeholder
    so the runtime can substitute the paragraph count.
    """

    pre_strategy: ClassVar[bool] = False

    def __init__(
        self,
        *,
        n_hypothetical: int = 1,
        timeout_seconds: float = 20.0,
        single_prompt: str | None = None,
        multi_prompt: str | None = None,
    ) -> None:
        if n_hypothetical < 1:
            raise ValueError(f"n_hypothetical must be >= 1; got {n_hypothetical}")
        self._n_hypothetical = n_hypothetical
        self._timeout_seconds = timeout_seconds
        self._single_prompt = single_prompt or DEFAULT_SINGLE_PROMPT
        self._multi_prompt = multi_prompt or DEFAULT_MULTI_PROMPT

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
                if self._n_hypothetical == 1:
                    return await self._single(query, chat_model)
                return await self._multiple(query, chat_model)
        except Exception as exc:
            # Broad catch — variation generation is best-effort.  The
            # consuming strategy always retains the original query, so
            # an empty list here just means "no fan-out, dense-only".
            logger.warning("[HyDETransformer] Failed: %s", exc)
            return []

    async def _single(self, query: str, chat_model: Any) -> list[str]:
        result = await chat_model.ainvoke(
            [
                SystemMessage(content=self._single_prompt),
                HumanMessage(content=query),
            ],
            temperature=0.3,
        )
        text = (result.content or "").strip()
        return [text] if text else []

    async def _multiple(self, query: str, chat_model: Any) -> list[str]:
        result = await chat_model.ainvoke(
            [
                SystemMessage(content=self._multi_prompt.format(n=self._n_hypothetical)),
                HumanMessage(content=query),
            ],
            temperature=0.5,
        )
        lines = [line.strip() for line in (result.content or "").split("\n") if line.strip()]
        return lines[: self._n_hypothetical]
