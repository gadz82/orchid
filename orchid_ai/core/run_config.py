"""
Auth as execution context — carried in the LangGraph ``RunnableConfig``,
never in the (checkpointed) graph state.

The graph state is *durable computation state* and is serialised by
persisting checkpointers (SQLite, PostgreSQL).  An :class:`OrchidAuthContext`
is a *credential*, not state: it is short-lived, rotates, and must never
land in a checkpoint at rest.  So auth travels out-of-band in
``config["configurable"]["auth_context"]`` — LangGraph delivers the
config to every node at runtime but does **not** serialise arbitrary
``configurable`` values into the checkpoint (only ``thread_id`` /
``checkpoint_ns`` / ``checkpoint_id`` are persisted).

This module uses ONLY stdlib types — safe for ``core/``.  ``config`` is
treated structurally as a ``Mapping`` so importing langchain's
``RunnableConfig`` type is unnecessary.

Usage
-----
Entry points (API routers, CLI, the invoker, the events runner) build a
config with :func:`with_auth` and pass it to ``graph.ainvoke`` /
``graph.astream``::

    config = with_auth(auth, thread_id=chat_id)
    await graph.ainvoke({"messages": [...], "chat_id": chat_id}, config=config)

Node callables receive ``config`` from LangGraph and read auth with
:func:`auth_from_config`::

    async def supervisor_node(state, config):
        auth = auth_from_config(config)
        ...
"""

from __future__ import annotations

from typing import Any, Mapping

from .state import OrchidAuthContext

__all__ = ["CONFIG_KEY_AUTH", "auth_from_config", "with_auth"]

#: ``config["configurable"]`` key under which the per-request
#: :class:`OrchidAuthContext` is carried.
CONFIG_KEY_AUTH = "auth_context"


def auth_from_config(config: Mapping[str, Any] | None) -> OrchidAuthContext | None:
    """Extract the :class:`OrchidAuthContext` from a ``RunnableConfig``.

    Returns ``None`` when no config is provided or no auth was injected
    (the same ``None``-tolerant contract callers had with
    ``state.get("auth_context")``).
    """
    if not config:
        return None
    configurable = config.get("configurable") or {}
    return configurable.get(CONFIG_KEY_AUTH)


def with_auth(
    auth: OrchidAuthContext | None,
    *,
    thread_id: str | None = None,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a ``RunnableConfig`` carrying *auth* (and optional *thread_id*).

    Merges onto *base* without mutating it.  ``auth=None`` / ``thread_id=None``
    are simply not written, so this is safe to call with partial inputs.
    """
    config: dict[str, Any] = dict(base or {})
    configurable: dict[str, Any] = dict(config.get("configurable") or {})
    if auth is not None:
        configurable[CONFIG_KEY_AUTH] = auth
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    config["configurable"] = configurable
    return config
