"""
Standalone helper functions for agent operations.

These functions contain the reusable logic that ``OrchidAgent`` methods
delegate to.  They can also be imported directly by custom agents or
tests without requiring a ``OrchidAgent`` instance.

Conversation history extraction, query reformulation, RAG retrieval,
and LLM summarization each live here as a pure async function.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .scopes import OrchidRAGScope
from .state import OrchidAgentState

logger = logging.getLogger(__name__)


# ── User query extraction ──────────────────────────────────────


def extract_user_query(state: OrchidAgentState) -> str:
    """Walk messages in reverse to find the last human message."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            return str(msg.content)
        if type(msg).__name__ == "HumanMessage":
            return str(msg.content)
    return ""


# ── Conversation history extraction ────────────────────────────


def extract_conversation_history(
    state: OrchidAgentState,
    *,
    max_turns: int = 10,
    max_chars: int | None = None,
    skip_prefixes: tuple[str, ...] = ("[Supervisor",),
    strip_prefixes: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Extract clean user/assistant pairs from graph state messages.

    Filters out supervisor routing noise, strips agent name tags,
    excludes the current query (last HumanMessage), and caps output
    to ``max_turns`` exchange pairs.
    """
    messages = state.get("messages", [])
    if not messages:
        return []

    history: list[dict[str, str]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()
        content = str(msg.content) if hasattr(msg, "content") else str(msg)

        if not content.strip():
            continue

        # Skip internal messages (e.g. supervisor routing)
        if any(content.startswith(prefix) for prefix in skip_prefixes):
            continue

        if msg_type in ("human", "humanmessage"):
            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars] + "…"
            history.append({"role": "user", "content": content})
        elif msg_type in ("ai", "aimessage"):
            for prefix in strip_prefixes:
                if content.startswith(prefix):
                    content = content[len(prefix) :]
                    break
            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars] + "…"
            history.append({"role": "assistant", "content": content})

    # Drop the last user message — it will be added separately as the current query
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    # Keep only the most recent turns to avoid exceeding context limits
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]

    return history


# ── Conversation history compression ───────────────────────────


async def compress_conversation_history(
    history: list[dict[str, str]],
    *,
    chat_model: Any,
    recent_turns: int = 10,
) -> list[dict[str, str]]:
    """Compress older history into a summary, keeping recent turns verbatim.

    When history fits within the window, returns it unchanged (no LLM call).
    On LLM failure, returns only the recent turns (no crash).
    """
    max_recent = recent_turns * 2
    if len(history) <= max_recent:
        return history

    older = history[:-max_recent]
    recent = history[-max_recent:]

    older_text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    prompt = (
        "Summarise the following conversation in one short paragraph. "
        "Focus on: key topics discussed, entities mentioned, actions taken, "
        "and any outstanding questions or requests.\n\n" + older_text
    )

    try:
        result = await chat_model.ainvoke(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        summary_text = result.content or ""
    except Exception as exc:
        logger.warning("History compression failed (%s), falling back to truncation", exc)
        return recent

    compressed: list[dict[str, str]] = [
        {"role": "assistant", "content": f"[Conversation summary]\n{summary_text.strip()}"},
    ]
    compressed.extend(recent)
    return compressed


# ── RAG retrieval ──────────────────────────────────────────────


async def fetch_rag_context(
    query: str,
    scope: OrchidRAGScope,
    *,
    reader: Any,
    namespace: str,
    k: int = 5,
    agent_name: str = "",
) -> list[dict[str, Any]]:
    """Retrieve relevant documents from the vector store.

    When ``parent_content`` is present in document metadata (Parent
    Document Retriever pattern), it is used as the content instead
    of the child chunk's ``page_content``.
    """
    try:
        results = await reader.retrieve(query=query, namespace=namespace, k=k, scope=scope)
        rag_docs = []
        for r in results:
            content = r.document.metadata.get("parent_content", r.document.page_content)
            rag_docs.append(
                {
                    "content": content,
                    "score": round(r.score, 3),
                    "metadata": {
                        mk: mv
                        for mk, mv in r.document.metadata.items()
                        if mk not in ("content", "embedding", "parent_content")
                    },
                }
            )
        logger.info("[%s] RAG retrieved %d docs (scope=%s)", agent_name, len(rag_docs), scope.tenant_id)
        return rag_docs
    except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
        logger.warning("[%s] RAG retrieval failed: %s", agent_name, exc)
        return []


# ── LLM summarization ─────────────────────────────────────────


async def summarise(
    query: str,
    mcp_data: dict[str, Any],
    rag_data: list[dict[str, Any]],
    *,
    system_prompt: str,
    chat_model: Any,
    temperature: float = 0.2,
    conversation_history: list[dict[str, str]] | None = None,
    prior_tool_context: dict[str, Any] | None = None,
    history_reminder: str | None = None,
    prior_results_header: str | None = None,
    rag_section_header: str | None = None,
    user_content_template: str | None = None,
    prior_results_max_chars: int = 4000,
) -> str:
    """Use LLM to produce a human-readable summary of RAG + MCP data.

    Standalone version of ``OrchidAgent.summarise()``.

    The four ``*_header`` / ``*_template`` / ``*_reminder`` overrides
    are forwarded by ``OrchidAgent.summarise()`` from the agent's
    :class:`OrchidAgentPromptConfig`.  ``None`` falls back to the
    module-level defaults defined in ``schema_prompts``.
    """
    # Local import — schema_prompts has zero external deps so this
    # stays cheap.  Importing at module top would create a circular
    # path through ``orchid_ai.config`` for callers that build their
    # own helpers.summarise() bridge.
    from ..config.schema_prompts import (
        DEFAULT_SUMMARISE_HISTORY_REMINDER,
        DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER,
        DEFAULT_SUMMARISE_RAG_HEADER,
        DEFAULT_SUMMARISE_USER_TEMPLATE,
    )

    history_reminder = history_reminder if history_reminder is not None else DEFAULT_SUMMARISE_HISTORY_REMINDER
    prior_results_header = (
        prior_results_header if prior_results_header is not None else DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER
    )
    rag_section_header = rag_section_header if rag_section_header is not None else DEFAULT_SUMMARISE_RAG_HEADER
    user_content_template = (
        user_content_template if user_content_template is not None else DEFAULT_SUMMARISE_USER_TEMPLATE
    )

    enriched_system = system_prompt

    if conversation_history:
        enriched_system += history_reminder

    if prior_tool_context:
        prior_json = json.dumps(prior_tool_context, indent=2, default=str)[:prior_results_max_chars]
        enriched_system += f"{prior_results_header}{prior_json}"

    rag_section = ""
    if rag_data:
        rag_section = rag_section_header + json.dumps(rag_data, indent=2, default=str) + "\n\n"

    user_content = user_content_template.format(
        query=query,
        rag_section=rag_section,
        mcp_data=json.dumps(mcp_data, indent=2, default=str),
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": enriched_system}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})

    result = await chat_model.ainvoke(messages, temperature=temperature)
    return result.content or ""
