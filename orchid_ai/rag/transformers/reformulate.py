"""
``ReformulateTransformer`` — agent-entry conversational rewrite.

Marked ``pre_strategy=True``: the rewritten query replaces the original
once at the start of the agent's turn and feeds **everything**
downstream (RAG retrieval, the agentic loop, summarisation).  Returns
exactly one query — the runtime check in
:func:`orchid_ai.core.retrieval.apply_pre_strategy` enforces that
contract.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ...core.retrieval import OrchidQueryTransformer

logger = logging.getLogger(__name__)


DEFAULT_REFORMULATE_PROMPT = (
    "You are a query reformulation assistant. Given the conversation history "
    "and the user's latest message, rewrite the message as a clear, standalone "
    "search query that can be used to search a database or menu.\n\n"
    "RULES:\n"
    "- Resolve pronouns and references ('it', 'that', 'the first one', 'yes')\n"
    "- Extract the core intent (what the user actually wants)\n"
    "- Keep it short and specific (under 20 words)\n"
    "- If the query is already clear and standalone, return it unchanged\n"
    "- Return ONLY the reformulated query, nothing else"
)


class ReformulateTransformer(OrchidQueryTransformer):
    """Rewrite a query into a standalone search query using conversation history.

    The system prompt is configurable via the ``system_prompt``
    constructor kwarg (or YAML
    ``rag.retrieval.transformer_prompts.reformulate``).  ``None``
    (default) keeps :data:`DEFAULT_REFORMULATE_PROMPT`.
    """

    pre_strategy: ClassVar[bool] = True

    def __init__(self, *, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or DEFAULT_REFORMULATE_PROMPT

    async def transform(
        self,
        query: str,
        *,
        chat_model: Any,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        # Without a chat model we can't reformulate — return the original
        # query unchanged so the pre-strategy "exactly 1 result" contract
        # still holds.
        if chat_model is None:
            return [query]

        # Without history there's nothing to disambiguate against — passing
        # a one-message conversation to the LLM costs latency for no gain.
        if not history:
            return [query]

        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self._system_prompt},
                *history,
                {"role": "user", "content": query},
            ]
            result = await chat_model.ainvoke(messages, temperature=0)
            reformulated = (result.content or "").strip()
            if reformulated and len(reformulated) < 200:
                logger.info(
                    "[ReformulateTransformer] '%s' -> '%s'",
                    query[:80],
                    reformulated[:80],
                )
                return [reformulated]
        except Exception as exc:
            logger.warning("[ReformulateTransformer] Failed: %s", exc)

        return [query]
