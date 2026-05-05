"""Mini-agent lifecycle events — emit + parse helpers.

The four ``mini_agent.*`` lifecycle events surface through a
piggyback ``SystemMessage`` whose ``additional_kwargs`` carries a
structured payload.  The streaming router in ``orchid-api`` extracts
these messages from the message stream and re-emits them as SSE
events (``data: {"type": "mini_agent.started", ...}``).

This is the single source of truth for the event wire shape.  Both
the emitting agents (decomposer hook, mini node, aggregator) and the
streaming router consume it.

Why ``additional_kwargs``?  LangChain's ``BaseMessage.additional_kwargs``
is the standard slot for non-standard metadata that travels with the
message — survives serialisation, checkpointing, and the
``add_messages`` reducer's deduplication.  Using it keeps the contract
purely additive: anything that doesn't read ``orchid_event`` simply
ignores it.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from langchain_core.messages import BaseMessage, SystemMessage


__all__ = [
    "MINI_AGENT_EVENT_KEY",
    "MiniAgentEventName",
    "make_event_message",
    "extract_event",
    "is_event_message",
]


# Key on ``additional_kwargs``.  Picked deliberately generic
# (``orchid_event``) so other event categories can land on the same
# transport without colliding.
MINI_AGENT_EVENT_KEY: Final[str] = "orchid_event"


MiniAgentEventName = Literal[
    "mini_agent.decomposed",
    "mini_agent.started",
    "mini_agent.finished",
    "mini_agent.aggregated",
]


def make_event_message(
    event_name: MiniAgentEventName,
    data: dict[str, Any],
) -> SystemMessage:
    """Build a ``SystemMessage`` carrying a mini-agent lifecycle event.

    ``content`` is empty by design — the message is metadata-only,
    invisible to the user-visible synthesis stream.  The streaming
    router strips these out via :func:`is_event_message`.

    Parameters
    ----------
    event_name
        One of the four ``mini_agent.*`` event names.  Validated as a
        ``Literal`` at type-check time; runtime accepts any string but
        consumers should treat unknown names as inert.
    data
        JSON-serialisable payload.  See spec §13 for per-event shapes.
    """
    return SystemMessage(
        content="",
        additional_kwargs={MINI_AGENT_EVENT_KEY: event_name, "data": dict(data)},
    )


def extract_event(msg: Any) -> tuple[str, dict[str, Any]] | None:
    """Return ``(event_name, data)`` if ``msg`` is a mini-agent event.

    Returns ``None`` for anything else — including normal
    ``SystemMessage`` instances, ``AIMessage``, tool messages, or
    plain dicts.  The streaming router uses this to filter the
    user-visible message stream and translate matching messages into
    SSE frames.
    """
    if not isinstance(msg, BaseMessage):
        return None
    extras = getattr(msg, "additional_kwargs", None)
    if not isinstance(extras, dict):
        return None
    name = extras.get(MINI_AGENT_EVENT_KEY)
    if not isinstance(name, str):
        return None
    data = extras.get("data")
    if not isinstance(data, dict):
        data = {}
    return name, data


def is_event_message(msg: Any) -> bool:
    """True iff ``msg`` carries a mini-agent lifecycle event.

    Convenience wrapper for callers that don't need the payload.
    """
    return extract_event(msg) is not None
