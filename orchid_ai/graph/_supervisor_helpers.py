"""Shared helpers used by the supervisor's routing, synthesis, and sequential-advance phases.

Extracted from :mod:`supervisor` when it was split into dedicated
:class:`SequentialAdvancer` and :class:`ResponseSynthesizer` collaborators,
so they have one home regardless of which collaborator imports them.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .state import GraphState


def _filter_internal_messages(
    messages: list[BaseMessage],
    *,
    skip_prefixes: tuple[str, ...] = ("[Supervisor",),
) -> list[BaseMessage]:
    """Remove internal routing messages (e.g. supervisor dispatches) from a message list."""
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            if any(content.startswith(prefix) for prefix in skip_prefixes):
                continue
        filtered.append(msg)
    return filtered


def _extract_single_agent_response(state: GraphState) -> str | None:
    """Return an agent's final text when exactly one agent ran this turn.

    Walks the messages added since the last ``HumanMessage`` and counts
    how many distinct ``[X Agent]\\n…`` outputs were produced.  Returns
    the (prefix-stripped) text only when:

    * exactly one agent message is present in the current turn,
    * its content is non-empty after stripping the prefix.

    Returns ``None`` in every other case — multi-agent fan-in,
    sequential pipelines, skill executions (multiple agents), or empty
    agent output — so synthesis still runs and merges results.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    # Find the index of the last user (human) message so we only look
    # at this turn's agent output.  ``HumanMessage`` covers both the
    # langchain-core class and the ``type == "human"`` duck-type used
    # in some test doubles.
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            last_user_idx = i
    if last_user_idx < 0:
        return None

    current_turn = messages[last_user_idx + 1 :]

    agent_outputs: list[str] = []
    for msg in current_turn:
        if not (isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai")):
            continue
        content = str(getattr(msg, "content", "") or "")
        if not content:
            continue
        # Skip supervisor internal messages (routing dispatch banners,
        # handoff prefixes) — those aren't agent outputs.
        if content.startswith("[Supervisor"):
            continue
        # Skip messages with pending tool calls — those aren't final answers.
        if getattr(msg, "tool_calls", None):
            continue
        # Strip the ``[Foo Agent]\n`` prefix added by ``GenericAgent.run``.
        # Only treat messages that carry that prefix as agent outputs;
        # bare AIMessages without a prefix could be from anywhere.
        if not (content.startswith("[") and "Agent]\n" in content[:80]):
            continue
        # Extract the text after the prefix
        newline_idx = content.find("\n")
        if newline_idx < 0:
            continue
        body = content[newline_idx + 1 :].strip()
        if body:
            agent_outputs.append(body)

    if len(agent_outputs) != 1:
        return None
    return agent_outputs[0]


def _to_llm_messages(
    system: str,
    state_messages: list[BaseMessage],
) -> list[dict[str, str]]:
    """Convert LangGraph messages to the [{role, content}] format used by BaseChatModel."""
    llm_msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    for msg in state_messages:
        if isinstance(msg, HumanMessage):
            llm_msgs.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            llm_msgs.append({"role": "assistant", "content": str(msg.content)})
    return llm_msgs


async def _llm_complete(
    chat_model: BaseChatModel | None,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    response_format: dict[str, str] | None = None,
) -> str:
    """Call LLM via the injected BaseChatModel."""
    if not chat_model:
        raise RuntimeError("Supervisor requires a BaseChatModel. Pass chat_model= when building the graph.")
    kwargs: dict = {"temperature": temperature}
    if response_format:
        kwargs["response_format"] = response_format
    result = await chat_model.ainvoke(messages, **kwargs)
    return result.content or ""
