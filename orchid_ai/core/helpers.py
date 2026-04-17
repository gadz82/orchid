"""
Standalone helper functions for agent operations.

These functions contain the reusable logic that ``BaseAgent`` methods
delegate to.  They can also be imported directly by custom agents or
tests without requiring a ``BaseAgent`` instance.

Conversation history extraction, query reformulation, RAG retrieval,
and LLM summarization each live here as a pure async function.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .scopes import RAGScope
from .state import AgentState

logger = logging.getLogger(__name__)


# ── User query extraction ──────────────────────────────────────


def extract_user_query(state: AgentState) -> str:
    """Walk messages in reverse to find the last human message."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            return str(msg.content)
        if type(msg).__name__ == "HumanMessage":
            return str(msg.content)
    return ""


# ── Conversation history extraction ────────────────────────────


def extract_conversation_history(
    state: AgentState,
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
    history: list[dict[str, str]] = []
    max_messages = max_turns * 2

    # Skip the last HumanMessage (current query — added separately)
    end_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, "type") and msg.type == "human":
            end_idx = i
            break
        if type(msg).__name__ == "HumanMessage":
            end_idx = i
            break

    for msg in messages[:end_idx]:
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue

        is_human = (hasattr(msg, "type") and msg.type == "human") or type(msg).__name__ == "HumanMessage"
        is_ai = (hasattr(msg, "type") and msg.type == "ai") or type(msg).__name__ == "AIMessage"

        if is_human:
            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "\u2026"
            history.append({"role": "user", "content": content})

        elif is_ai:
            # Skip internal supervisor messages
            if any(content.startswith(prefix) for prefix in skip_prefixes):
                continue
            # Strip agent name prefixes from content
            for prefix in strip_prefixes:
                if content.startswith(prefix):
                    content = content[len(prefix) :].strip()
                    break
            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "\u2026"
            history.append({"role": "assistant", "content": content})

    return history[-max_messages:]


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
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
        logger.warning("History compression failed (%s), falling back to truncation", exc)
        return recent

    compressed: list[dict[str, str]] = [
        {"role": "assistant", "content": f"[Conversation summary]\n{summary_text.strip()}"},
    ]
    compressed.extend(recent)
    return compressed


# ── Query reformulation ────────────────────────────────────────

_REFORMULATE_PROMPT = (
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


async def reformulate_query(
    query: str,
    state: AgentState,
    *,
    chat_model: Any,
    agent_name: str = "",
) -> str:
    """Rewrite the user's query as a standalone search query using conversation history.

    Returns the original query unchanged on failure or when no history exists.
    """
    if not chat_model:
        return query

    history = extract_conversation_history(state, max_turns=5, max_chars=500)
    if not history:
        return query

    try:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _REFORMULATE_PROMPT},
            *history,
            {"role": "user", "content": query},
        ]

        result = await chat_model.ainvoke(messages, temperature=0)
        reformulated = (result.content or "").strip()

        if reformulated and len(reformulated) < 200:
            logger.info("[%s] Query reformulated: '%s' -> '%s'", agent_name, query[:80], reformulated[:80])
            return reformulated
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
        logger.warning("[%s] Query reformulation failed: %s", agent_name, exc)

    return query


# ── RAG retrieval ──────────────────────────────────────────────


async def fetch_rag_context(
    query: str,
    scope: RAGScope,
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
) -> str:
    """Use LLM to produce a human-readable summary of RAG + MCP data.

    Standalone version of ``BaseAgent.summarise()``.
    """
    enriched_system = system_prompt

    if conversation_history:
        enriched_system += (
            "\n\nIMPORTANT: The conversation history below shows prior exchanges. "
            "Always focus on the user's LATEST message and its relationship to "
            "the most recent topic. Do NOT change topic or introduce unrelated "
            "content unless the user explicitly asks for something new."
        )

    if prior_tool_context:
        prior_json = json.dumps(prior_tool_context, indent=2, default=str)[:4000]
        enriched_system += f"\n\n--- Previous Tool Results (from prior turns) ---\n{prior_json}"

    rag_section = ""
    if rag_data:
        rag_section = "Background knowledge (from RAG):\n" + json.dumps(rag_data, indent=2, default=str) + "\n\n"

    user_content = (
        f"User query: {query}\n\n{rag_section}Live data (from API):\n{json.dumps(mcp_data, indent=2, default=str)}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": enriched_system}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})

    result = await chat_model.ainvoke(messages, temperature=temperature)
    return result.content or ""
