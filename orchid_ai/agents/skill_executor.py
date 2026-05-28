"""Skill execution — runs multi-step agent-level skills."""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Awaitable, Callable

from ..config.schema import OrchidAgentSkillStepConfig
from ..config.tool_registry import filter_to_schema, get_tool
from ..core.tool import OrchidToolInput
from ..core.state import OrchidAuthContext

logger = logging.getLogger(__name__)

MAX_AGENT_SKILL_DEPTH = 3

# Task-local recursion depth.  Replaces the previous
# ``self._skill_depth`` instance attribute, which was mutated on the
# *peer* executor (``peer._skill_executor._skill_depth = ...``) — a
# pattern that races under LangGraph parallel fan-out and clobbered
# the depth back to ``0`` (not the previous value) on unwind.  The
# ContextVar is asyncio-task-local, so concurrent skill chains never
# share counter state.
_skill_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "orchid.skill_depth",
    default=0,
)


class SkillExecutor:
    """Executes agent-level skill steps (MCP tools, built-in tools, or agent invocations)."""

    def __init__(
        self,
        *,
        agent_name: str,
        mcp_dispatcher: Any,
        builtin_tool_caller: Callable[..., Awaitable[Any]],
        agent_peers: dict[str, Any] | None = None,
        content_sources: Any = None,
    ):
        self._agent_name = agent_name
        self._mcp_dispatcher = mcp_dispatcher
        self._builtin_tool_caller = builtin_tool_caller
        self._agent_peers = agent_peers or {}
        self._content_sources = content_sources

    @property
    def _skill_depth(self) -> int:
        """Current task-local recursion depth.

        Read-only proxy over :data:`_skill_depth_var`.  Kept as a
        property (not an instance attribute) so legacy callers that
        peeked at the field still see the right value, while
        concurrent skill chains cannot stomp on each other's
        counter.  Use the Token API in :meth:`_run_agent_step` to
        increment / restore.
        """
        return _skill_depth_var.get()

    async def run_skill(
        self,
        skill_name: str,
        steps: list[OrchidAgentSkillStepConfig],
        query: str,
        auth: OrchidAuthContext,
    ) -> dict[str, Any]:
        """Execute a named skill step-by-step."""
        results: dict[str, Any] = {}
        previous_results: dict[str, Any] = {}
        for step in steps:
            key = step.step_key
            step_result = await self._run_step(step, query, auth, previous_results)
            results[key] = step_result
            previous_results[key] = step_result
        return results

    async def _run_step(
        self,
        step: OrchidAgentSkillStepConfig,
        query: str,
        auth: OrchidAuthContext,
        previous_results: dict[str, Any],
    ) -> Any:
        """Execute a single skill step."""
        step_name = step.step_key
        try:
            if step.agent:
                return await self._run_agent_step(
                    step.agent,
                    step.instruction,
                    query,
                    auth,
                    previous_results,
                )
            elif step.source and step.source != "builtin":
                return await self._mcp_dispatcher.call_tool_by_source(
                    step.source,
                    step.tool,
                    query,
                    auth,
                    step.arguments,
                    previous_results,
                )
            else:
                return await self._run_builtin_step(
                    step.tool,
                    query,
                    auth,
                    step.arguments,
                    previous_results,
                    content_sources=self._content_sources,
                )
        except (ValueError, TypeError, KeyError, RuntimeError, ConnectionError, TimeoutError, OSError) as exc:
            logger.error("[%s] Skill step '%s' failed: %s", self._agent_name, step_name, exc)
            return f"error: {exc}"

    async def _run_builtin_step(
        self,
        tool_name: str,
        query: str,
        auth: OrchidAuthContext,
        step_arguments: dict[str, Any],
        previous_results: dict[str, Any],
        content_sources: Any = None,
    ) -> Any:
        """Execute a built-in tool skill step."""
        tool = get_tool(tool_name)
        params = filter_to_schema(step_arguments, tool.get_parameters_schema())
        logger.debug(
            "[%s] Builtin step '%s': schema params=%s, provided=%s",
            self._agent_name,
            tool_name,
            sorted(tool.get_parameters_schema().get("properties", {}).keys()),
            sorted(params.keys()),
        )

        tool_input = OrchidToolInput(
            parameters=params,
            query=query,
            context=previous_results or None,
            auth_context=auth,
            content_sources=content_sources,
        )
        output = await tool.invoke(tool_input)
        return output.result

    async def _run_agent_step(
        self,
        agent_name: str,
        instruction: str,
        query: str,
        auth: OrchidAuthContext,
        previous_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke another agent within a skill step."""
        from langchain_core.messages import HumanMessage

        if agent_name not in self._agent_peers:
            available = list(self._agent_peers.keys())
            raise ValueError(f"Agent '{agent_name}' not available. Available peers: {available}")

        current_depth = _skill_depth_var.get()
        if current_depth >= MAX_AGENT_SKILL_DEPTH:
            raise RecursionError(
                f"Agent skill depth exceeded ({current_depth}). "
                f"'{self._agent_name}' tried to invoke '{agent_name}' but max depth of "
                f"{MAX_AGENT_SKILL_DEPTH} reached."
            )

        peer = self._agent_peers[agent_name]
        effective_query = instruction or query
        if previous_results:
            context_str = json.dumps(previous_results, indent=2, default=str)
            effective_query += f"\n\nContext from previous steps:\n```json\n{context_str}\n```"

        from ..core.state import OrchidAgentState

        mini_state: OrchidAgentState = {
            "messages": [HumanMessage(content=effective_query)],
            "auth_context": auth,
            "chat_id": "",
            "mcp_context": {},
        }

        # Increment task-local depth via the ContextVar.  The Token
        # returned by ``set`` is reset in ``finally`` so the prior
        # depth (not ``0``) is restored — fixes the pre-existing
        # bug where two-level nesting unwound to ``0`` mid-chain.
        depth_token = _skill_depth_var.set(current_depth + 1)
        try:
            logger.info(
                "[%s] Invoking peer '%s' (depth=%d): %s",
                self._agent_name,
                agent_name,
                current_depth + 1,
                effective_query[:120],
            )
            result_state = await peer.run(mini_state)
        finally:
            _skill_depth_var.reset(depth_token)

        mcp_data = result_state.get("mcp_context", {})
        messages = result_state.get("messages", [])
        response_text = messages[0].content if messages else ""
        return {
            "agent": agent_name,
            "data": mcp_data.get(agent_name, mcp_data),
            "response": response_text,
        }
