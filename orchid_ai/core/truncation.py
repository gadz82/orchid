"""
Message truncation strategies.

When conversation messages exceed ``max_chars``, these strategies
determine how to shorten them without entirely losing the content.

This module lives in ``core/`` so it can be used by helpers without
external dependencies.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrchidTruncationStrategy(str, Enum):
    """How to handle messages that exceed max_chars."""

    HARD = "hard"
    MIDDLE = "middle"
    LLM = "llm"
    SEMANTIC = "semantic"


_TRUNCATION_MARKER = "…[truncated]…"


def truncate_content(
    content: str,
    max_chars: int,
    strategy: OrchidTruncationStrategy = OrchidTruncationStrategy.HARD,
) -> str:
    """Truncate a message content string according to the given strategy.

    LLM and SEMANTIC strategies fall back to MIDDLE when called
    synchronously (they need async context or embeddings respectively).
    """
    if len(content) <= max_chars:
        return content

    if strategy == OrchidTruncationStrategy.HARD:
        return content[: max_chars - 1] + "…"

    if strategy == OrchidTruncationStrategy.MIDDLE:
        return _truncate_middle(content, max_chars)

    return _truncate_middle(content, max_chars)


async def truncate_content_async(
    content: str,
    max_chars: int,
    strategy: OrchidTruncationStrategy = OrchidTruncationStrategy.HARD,
    chat_model: Any = None,
    query: str | None = None,
) -> str:
    """Async version that supports LLM and SEMANTIC strategies."""
    if len(content) <= max_chars:
        return content

    if strategy == OrchidTruncationStrategy.HARD:
        return content[: max_chars - 1] + "…"

    if strategy == OrchidTruncationStrategy.MIDDLE:
        return _truncate_middle(content, max_chars)

    if strategy == OrchidTruncationStrategy.LLM:
        if not chat_model:
            return _truncate_middle(content, max_chars)
        try:
            return await _truncate_llm(content, max_chars, chat_model)
        except Exception:
            logger.warning("LLM truncation failed, falling back to MIDDLE")
            return _truncate_middle(content, max_chars)

    if strategy == OrchidTruncationStrategy.SEMANTIC:
        return _truncate_middle(content, max_chars)

    return _truncate_middle(content, max_chars)


def _truncate_middle(content: str, max_chars: int) -> str:
    """Keep first 40% and last 40% with a truncation marker in between."""
    if len(content) <= max_chars:
        return content

    head_size = int(max_chars * 0.4)
    tail_size = int(max_chars * 0.4)
    marker = _TRUNCATION_MARKER
    remaining = max_chars - head_size - tail_size - len(marker)

    if remaining < 0:
        return content[: max_chars - 1] + "…"

    return content[:head_size] + marker + content[-tail_size:]


async def _truncate_llm(content: str, max_chars: int, chat_model: Any) -> str:
    """Ask the LLM to produce a concise version of the message."""
    prompt = (
        "Summarise the following text concisely in no more than "
        f"{max_chars} characters. Preserve key facts, numbers, "
        "and names. Do not add commentary.\n\n{text}"
    ).format(text=content)

    result = await chat_model.ainvoke(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    summary = result.content or ""

    if len(summary) > max_chars:
        return summary[:max_chars]

    return summary
