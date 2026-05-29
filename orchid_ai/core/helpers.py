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

from .memory_types import (
    OrchidConversationSummary,
    DEFAULT_NARRATIVE_FALLBACK_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT,
)
from .scopes import OrchidRAGScope
from .state import OrchidAgentState
from .truncation import OrchidTruncationStrategy, truncate_content

logger = logging.getLogger(__name__)


# ── User query extraction ──────────────────────────────────────


def extract_user_query(state: OrchidAgentState) -> str:
    """Walk messages in reverse to find the last human message.

    Uses a single canonical duck-type check via
    ``getattr(msg, "type", None) == "human"`` which covers both
    LangChain ``HumanMessage`` (which sets ``.type = "human"``) and
    any compatible message object.  The ``isinstance`` check is
    intentionally avoided here to keep ``core/`` free of
    ``langchain_core`` imports.
    """
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", None) == "human":
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
    truncation_strategy: str = "hard",
) -> list[dict[str, str]]:
    """Extract clean user/assistant pairs from graph state messages.

    Filters out supervisor routing noise, strips agent name tags,
    excludes the current query (last HumanMessage), and caps output
    to ``max_turns`` exchange pairs.

    When ``max_chars`` is set, individual messages longer than
    ``max_chars`` are truncated using ``truncation_strategy``
    (one of ``"hard"``, ``"middle"``, ``"llm"``, ``"semantic"``).

    Iterates from the end and breaks early once ``max_turns`` pairs
    are collected, avoiding O(n) full-list construction for long
    histories (1000+ messages).
    """
    strategy = OrchidTruncationStrategy(truncation_strategy)
    messages = state.get("messages", [])
    if not messages:
        return []

    max_messages = max_turns * 2
    # Iterate from end, skip the last human message only if it's the
    # current unanswered query (i.e., no AI message follows it).
    history: list[dict[str, str]] = []
    skip_last_human = False

    # Determine if the last relevant message is a human message (current query)
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()
        content = str(msg.content) if hasattr(msg, "content") else str(msg)
        if not content.strip():
            continue
        if any(content.startswith(prefix) for prefix in skip_prefixes):
            continue
        if msg_type in ("human", "humanmessage"):
            skip_last_human = True
        break

    skipped = False
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()
        content = str(msg.content) if hasattr(msg, "content") else str(msg)

        if not content.strip():
            continue

        if any(content.startswith(prefix) for prefix in skip_prefixes):
            continue

        if msg_type in ("human", "humanmessage"):
            if skip_last_human and not skipped:
                skipped = True
                continue
            if max_chars is not None and len(content) > max_chars:
                content = truncate_content(content, max_chars, strategy)
            history.append({"role": "user", "content": content})
        elif msg_type in ("ai", "aimessage"):
            for prefix in strip_prefixes:
                if content.startswith(prefix):
                    content = content[len(prefix) :]
                    break
            if max_chars is not None and len(content) > max_chars:
                content = truncate_content(content, max_chars, strategy)
            history.append({"role": "assistant", "content": content})

        if len(history) >= max_messages:
            break

    history.reverse()
    return history


# ── Conversation history compression ───────────────────────────


async def compress_conversation_history(
    history: list[dict[str, str]],
    *,
    chat_model: Any,
    recent_turns: int = 10,
    running_summary: str | None = None,
    structured_output: bool = False,
    compression_system_prompt: str | None = None,
    compression_user_prompt: str | None = None,
    extension_user_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Compress older history into a summary, keeping recent turns verbatim.

    When history fits within the window, returns it unchanged (no LLM call).
    On LLM failure, returns only the recent turns (no crash).

    When ``running_summary`` is provided, older messages are used to
    **extend** the existing summary incrementally rather than
    summarizing them from scratch.  This is the stateful running-summary
    pattern that avoids O(n^2) re-computation.

    When ``structured_output`` is ``True``, the LLM is prompted to
    produce a structured JSON summary with entity extraction.  On JSON
    parse failure the system falls back to a narrative-only summary.

    Prompt overrides (``*_prompt``) use ``None`` → module-level defaults.

    PII / data residency responsibility
    ------------------------------------
    This function sends conversation content to an LLM for summarisation.
    The framework does NOT automatically apply PII redaction before the
    LLM call.  For multi-tenant deployments with strict data residency
    or PII requirements, integrators are responsible for either:

    1. Applying PII guardrails (``orchid_ai.guardrails``) to messages
       BEFORE they enter the conversation history, or
    2. Wrapping this function with a pre-processing step that redacts
       sensitive fields before summarisation, or
    3. Using a self-hosted LLM that satisfies data residency constraints.

    The ``chat_model`` used here should be configured to route through
    an LLM endpoint that meets the deployment's compliance requirements.
    """
    max_recent = recent_turns * 2
    if len(history) <= max_recent:
        return history

    older = history[:-max_recent]
    recent = history[-max_recent:]

    older_text = "\n".join(f"{m['role']}: {m['content']}" for m in older)

    if running_summary:
        if structured_output:
            system_prompt = (
                "You are a conversation summarizer that produces structured summaries. "
                "You have an existing summary and new messages to incorporate. "
                "Update the summary to reflect new information, remove contradicted "
                "facts, and merge duplicate entities.\n\n"
                "Output ONLY valid JSON."
            )
            user_prompt = (
                extension_user_prompt
                or (
                    "Given the existing summary below and the new conversation messages, "
                    "produce an updated summary that incorporates all new information.\n\n"
                    "Existing summary:\n{existing_summary}\n\n"
                    "New messages:\n{new_messages}"
                )
            ).format(existing_summary=running_summary, new_messages=older_text)
        else:
            system_prompt = compression_system_prompt or "You are a conversation summarizer. Output only the summary."
            user_prompt = (
                extension_user_prompt
                or (
                    "Given this existing summary and these new conversation messages, "
                    "produce an updated summary that incorporates all new information.\n\n"
                    "Existing summary:\n{existing_summary}\n\n"
                    "New messages:\n{new_messages}"
                )
            ).format(existing_summary=running_summary, new_messages=older_text)
    elif structured_output:
        system_prompt = compression_system_prompt or DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT
        user_prompt = (compression_user_prompt or DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT).format(transcript=older_text)
    else:
        system_prompt = compression_system_prompt or "You are a conversation summarizer. Output only the summary."
        user_prompt = (
            compression_user_prompt
            or (
                "Summarise the following conversation in one short paragraph. "
                "Focus on: key topics discussed, entities mentioned, actions taken, "
                "and any outstanding questions or requests.\n\n{transcript}"
            )
        ).format(transcript=older_text)

    try:
        result = await chat_model.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        summary_text = result.content or ""
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("History compression failed (%s), falling back to truncation", exc)
        return recent

    if structured_output:
        structured = OrchidConversationSummary.from_json(summary_text)
        if structured is not None:
            context_string = structured.to_context_string()
        else:
            logger.warning("Structured summary JSON parse failed, falling back to narrative")
            context_string = _narrative_fallback(older_text)
    else:
        context_string = summary_text.strip()

    compressed: list[dict[str, str]] = [
        {"role": "assistant", "content": f"[Conversation summary]\n{context_string}"},
    ]
    compressed.extend(recent)
    return compressed


def _narrative_fallback(transcript: str) -> str:
    return DEFAULT_NARRATIVE_FALLBACK_PROMPT.format(transcript=transcript)


def filter_summary_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only the non-summary messages from *history*.

    The ``[Conversation summary]`` prefix is an internal compression
    artefact — it should not be fed back into the summariser as if it
    were a real conversation turn.
    """
    return [m for m in history if not m.get("content", "").startswith("[Conversation summary]")]


# Backward-compat alias (was private, now public)
_filter_summary_messages = filter_summary_messages


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
