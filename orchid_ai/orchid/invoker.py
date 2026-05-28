"""OrchidInvoker — invoke / stream / resume, including persistence.

M1 refactoring: extracted from the 1281-LOC ``Orchid`` god module.
Owns the three call shapes and the shared invocation prelude.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.persistence.base import OrchidChatStorage

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────


@dataclass
class OrchidPendingApproval:
    """A single tool-approval request surfaced by ``interrupt()``."""

    tool: str
    args: dict[str, Any]
    agent: str
    interrupt_id: str


@dataclass
class OrchidInvokeResult:
    """Result of a single :meth:`OrchidInvoker.invoke` call."""

    response: str
    chat_id: str
    agents_used: list[str] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)
    interrupted: bool = False
    approvals_needed: list[OrchidPendingApproval] = field(default_factory=list)
    mcp_context: dict[str, Any] = field(default_factory=dict)
    rag_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedInvocation:
    """Return type of :meth:`OrchidInvoker._prepare_invocation`."""

    auth_ctx: OrchidAuthContext
    chat_id: str
    state: dict[str, Any]
    graph_config: dict[str, Any]


# ── OrchidInvoker ──────────────────────────────────────────────


class OrchidInvoker:
    """Handles the three call shapes: invoke, stream, resume.

    Requires a compiled graph and an optional chat storage backend.
    The caller is responsible for lifecycle (close) — this class only
    owns invocation logic.
    """

    def __init__(
        self,
        graph: Any,  # CompiledGraph
        chat_repo: OrchidChatStorage | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._graph = graph
        self._chat_repo = chat_repo
        self._checkpointer = checkpointer

    async def invoke(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: OrchidAuthContext | None = None,
        history: list[BaseMessage] | None = None,
        persist: bool = True,
    ) -> OrchidInvokeResult:
        """Run a single request through the agent graph."""
        prepared = await self._prepare_invocation(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            persist=persist,
        )

        try:
            result = await self._graph.ainvoke(prepared.state, config=prepared.graph_config)
        except GraphInterrupt as exc:
            return self._interrupt_to_result(exc, prepared.chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(prepared.chat_id, "user", message)
            await self._chat_repo.add_message(
                prepared.chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return self._result_from_graph_output(result, prepared.chat_id)

    async def resume(
        self,
        chat_id: str,
        *,
        approved: bool = True,
        persist: bool = True,
    ) -> OrchidInvokeResult:
        """Continue a graph previously paused by ``interrupt()``."""
        if self._checkpointer is None:
            raise RuntimeError(
                "Cannot resume without a checkpointer — "
                "construct the client with checkpointer_type='memory' (or 'sqlite'/'postgres')."
            )

        graph_config = {"configurable": {"thread_id": chat_id}}

        try:
            result = await self._graph.ainvoke(
                Command(resume={"approved": approved}),
                config=graph_config,
            )
        except GraphInterrupt as exc:
            return self._interrupt_to_result(exc, chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(
                chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return self._result_from_graph_output(result, chat_id)

    async def stream(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: OrchidAuthContext | None = None,
        history: list[BaseMessage] | None = None,
        stream_mode: str | list[str] = "updates",
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream graph events as an async iterator."""
        prepared = await self._prepare_invocation(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            persist=False,
        )

        async for mode, chunk in self._graph.astream(
            prepared.state,
            config=prepared.graph_config,
            stream_mode=stream_mode if isinstance(stream_mode, list) else [stream_mode],
        ):
            yield mode, chunk

    # ── Internal helpers ────────────────────────────────────

    async def _resolve_history(
        self,
        chat_id: str,
        explicit_history: list[BaseMessage] | None,
    ) -> list[BaseMessage]:
        """Resolve conversation history for the next invocation."""
        if self._checkpointer is not None:
            return []

        if explicit_history is not None:
            return list(explicit_history)

        if self._chat_repo is None:
            return []

        rows = await self._chat_repo.get_messages(chat_id, limit=50)
        out: list[BaseMessage] = []
        for row in rows:
            if row.role == "user":
                out.append(HumanMessage(content=row.content, id=row.id))
            elif row.role == "assistant":
                out.append(AIMessage(content=row.content, id=row.id))
        return out

    async def _prepare_invocation(
        self,
        *,
        message: str,
        chat_id: str | None,
        user_id: str,
        tenant_id: str,
        access_token: str,
        auth: OrchidAuthContext | None,
        history: list[BaseMessage] | None,
        persist: bool,
    ) -> _PreparedInvocation:
        """Assemble everything ``graph.ainvoke`` / ``astream`` need."""
        auth_ctx = auth or OrchidAuthContext(
            access_token=access_token,
            tenant_key=tenant_id,
            user_id=user_id,
        )

        if chat_id is None and persist and self._chat_repo is not None:
            new_chat = await self._chat_repo.create_chat(
                tenant_id=auth_ctx.tenant_key,
                user_id=auth_ctx.user_id,
                title=message[:50],
            )
            effective_chat_id = new_chat.id
        else:
            effective_chat_id = chat_id or str(uuid.uuid4())

        resolved_history = await self._resolve_history(effective_chat_id, history)

        new_user_msg = HumanMessage(content=message)
        state: dict[str, Any] = {
            "messages": [*resolved_history, new_user_msg],
            "auth_context": auth_ctx,
            "chat_id": effective_chat_id,
        }
        graph_config = {"configurable": {"thread_id": effective_chat_id}}

        return _PreparedInvocation(
            auth_ctx=auth_ctx,
            chat_id=effective_chat_id,
            state=state,
            graph_config=graph_config,
        )

    def _result_from_graph_output(self, result: dict[str, Any], chat_id: str) -> OrchidInvokeResult:
        """Build an :class:`OrchidInvokeResult` from the graph's return payload."""
        return OrchidInvokeResult(
            response=result.get("final_response", ""),
            chat_id=chat_id,
            agents_used=list(result.get("active_agents") or []),
            messages=list(result.get("messages") or []),
            mcp_context=dict(result.get("mcp_context") or {}),
            rag_context=dict(result.get("rag_context") or {}),
        )

    @staticmethod
    def _interrupt_to_result(exc: GraphInterrupt, chat_id: str) -> OrchidInvokeResult:
        """Convert a ``GraphInterrupt`` into an :class:`OrchidInvokeResult`."""
        interrupts = exc.args[0] if exc.args else []
        approvals: list[OrchidPendingApproval] = []
        for i in interrupts:
            val = getattr(i, "value", None)
            if isinstance(val, dict):
                approvals.append(
                    OrchidPendingApproval(
                        tool=val.get("tool", ""),
                        args=val.get("args", {}),
                        agent=val.get("agent", ""),
                        interrupt_id=str(getattr(i, "id", "")),
                    )
                )
            else:
                approvals.append(
                    OrchidPendingApproval(
                        tool=str(val) if val is not None else "",
                        args={},
                        agent="",
                        interrupt_id=str(getattr(i, "id", "")),
                    )
                )
        return OrchidInvokeResult(
            response="",
            chat_id=chat_id,
            interrupted=True,
            approvals_needed=approvals,
        )
