"""
Conversation memory implementations.

Provides concrete memory strategies that implement the
``OrchidConversationMemory`` ABC from ``core/memory.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..core.memory import OrchidConversationMemory
from ..core.memory_types import (
    DEFAULT_STRUCTURED_EXTENSION_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_EXTENSION_USER_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT,
    OrchidConversationSummary,
)

logger = logging.getLogger(__name__)


class OrchidInMemoryConversationMemory(OrchidConversationMemory):
    """Stores running summaries in ``OrchidChatStorage`` (SQLite or PostgreSQL).

    Uses the chat storage backend to persist summaries between
    invocations.  The LLM call for summary extension uses the
    injected ``chat_model`` (typically a cheap/fast model like
    ``gemini/gemini-2.5-flash-lite``).

    When ``structured_output`` is ``True`` (default), summaries are
    stored as JSON with entity extraction.  The LLM is prompted to
    produce structured output; on JSON parse failure the system
    falls back to a narrative-only summary.
    """

    def __init__(self, chat_storage: Any, chat_model: Any, *, structured_output: bool = True):
        self._storage = chat_storage
        self._chat_model = chat_model
        self._structured_output = structured_output

    async def get_running_summary(self, chat_id: str) -> str | None:
        return await self._storage.get_conversation_summary(chat_id)

    async def update_running_summary(
        self,
        chat_id: str,
        new_messages: list[dict[str, str]],
        existing_summary: str | None,
    ) -> str:
        if not new_messages:
            return existing_summary or ""

        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in new_messages)
        turn_number = len(new_messages)

        if self._structured_output:
            return await self._update_structured(chat_id, transcript, existing_summary, turn_number)
        else:
            return await self._update_narrative(chat_id, transcript, existing_summary, turn_number)

    async def _update_narrative(
        self, chat_id: str, transcript: str, existing_summary: str | None, turn_number: int
    ) -> str:
        if existing_summary:
            prompt = (
                "Given this existing summary and these new conversation messages, "
                "produce an updated summary that incorporates all new information.\n\n"
                f"Existing summary:\n{existing_summary}\n\n"
                f"New messages:\n{transcript}"
            )
        else:
            prompt = (
                "Summarise the following conversation in one short paragraph. "
                "Focus on: key topics discussed, entities mentioned, actions taken, "
                "and any outstanding questions or requests.\n\n" + transcript
            )

        try:
            result = await self._chat_model.ainvoke(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            updated_summary = result.content or ""
        except Exception as exc:
            logger.warning(
                "Conversation memory update failed (%s), keeping existing summary",
                exc,
            )
            return existing_summary or ""

        await self._storage.save_conversation_summary(chat_id, updated_summary, turn_number)
        return updated_summary

    async def _update_structured(
        self, chat_id: str, transcript: str, existing_summary: str | None, turn_number: int
    ) -> str:
        if existing_summary:
            existing_parsed = OrchidConversationSummary.from_string_or_json(existing_summary)
            system_prompt = DEFAULT_STRUCTURED_EXTENSION_SYSTEM_PROMPT
            user_prompt = DEFAULT_STRUCTURED_EXTENSION_USER_PROMPT.format(
                existing_summary=json.dumps(existing_parsed.to_dict(), indent=2),
                new_messages=transcript,
            )
        else:
            system_prompt = DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT
            user_prompt = DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT.format(transcript=transcript)

        try:
            result = await self._chat_model.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            response_text = result.content or ""
        except Exception as exc:
            logger.warning(
                "Conversation memory update failed (%s), keeping existing summary",
                exc,
            )
            return existing_summary or ""

        parsed = OrchidConversationSummary.from_json(response_text)
        if parsed is not None:
            if existing_summary:
                existing_parsed = OrchidConversationSummary.from_string_or_json(existing_summary)
                merged = OrchidConversationSummary.merge(existing_parsed, parsed.to_dict())
                updated_summary = json.dumps(merged.to_dict())
            else:
                updated_summary = json.dumps(parsed.to_dict())
        else:
            logger.warning("Structured summary JSON parse failed, storing raw response")
            updated_summary = response_text

        await self._storage.save_conversation_summary(chat_id, updated_summary, turn_number)
        return updated_summary

    async def get_relevant_history(
        self,
        query: str,
        chat_id: str,
        k: int = 5,
    ) -> list[dict[str, str]]:
        return []

    async def store_conversation_turn(
        self,
        chat_id: str,
        tenant_id: str,
        user_id: str,
        turn: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass
