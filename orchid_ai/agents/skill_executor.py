"""Skill execution — runs multi-step agent-level skills."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from ..config.schema import AgentSkillStepConfig
from ..core.state import AuthContext

logger = logging.getLogger(__name__)

MAX_AGENT_SKILL_DEPTH = 3


class SkillExecutor:
    """Executes agent-level skill steps (MCP tools, built-in tools, or agent invocations)."""

    def __init__(
        self,
        *,
        agent_name: str,
        mcp_dispatcher: Any,
        builtin_tool_caller: Callable[..., Awaitable[Any]],
        agent_peers: dict[str, Any] | None = None,
    ):
        self._agent_name = agent_name
        self._mcp_dispatcher = mcp_dispatcher
        self._builtin_tool_caller = builtin_tool_caller
        self._agent_peers = agent_peers or {}
        self._skill_depth: int = 0

    async def run_skill(
        self,
        skill_name: str,
        steps: list[AgentSkillStepConfig],
        query: str,
        auth: AuthContext,
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
        step: AgentSkillStepConfig,
        query: str,
        auth: AuthContext,
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
                args = {"query": query, **step.arguments}
                if previous_results:
                    args["context"] = previous_results
                return await self._builtin_tool_caller(step.tool, **args)
        except (ValueError, TypeError, KeyError, RuntimeError, ConnectionError, TimeoutError, OSError) as exc:
            logger.error("[%s] Skill step '%s' failed: %s", self._agent_name, step_name, exc)
            return f"error: {exc}"

    async def _run_agent_step(
        self,
        agent_name: str,
        instruction: str,
        query: str,
        auth: AuthContext,
        previous_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke another agent within a skill step."""
        from langchain_core.messages import HumanMessage

        if agent_name not in self._agent_peers:
            available = list(self._agent_peers.keys())
            raise ValueError(f"Agent '{agent_name}' not available. Available peers: {available}")

        if self._skill_depth >= MAX_AGENT_SKILL_DEPTH:
            raise RecursionError(
                f"Agent skill depth exceeded ({self._skill_depth}). "
                f"'{self._agent_name}' tried to invoke '{agent_name}' but max depth of "
                f"{MAX_AGENT_SKILL_DEPTH} reached."
            )

        peer = self._agent_peers[agent_name]
        effective_query = instruction or query
        if previous_results:
            context_str = json.dumps(previous_results, indent=2, default=str)
            effective_query += f"\n\nContext from previous steps:\n```json\n{context_str}\n```"

        from ..core.state import AgentState

        mini_state: AgentState = {
            "messages": [HumanMessage(content=effective_query)],
            "auth_context": auth,
            "chat_id": "",
            "mcp_context": {},
        }

        # Track recursion depth on the peer if it has a skill executor
        if hasattr(peer, "_skill_executor") and peer._skill_executor:
            peer._skill_executor._skill_depth = self._skill_depth + 1

        try:
            logger.info(
                "[%s] Invoking peer '%s' (depth=%d): %s",
                self._agent_name,
                agent_name,
                self._skill_depth + 1,
                effective_query[:120],
            )
            result_state = await peer.run(mini_state)
        finally:
            if hasattr(peer, "_skill_executor") and peer._skill_executor:
                peer._skill_executor._skill_depth = 0

        mcp_data = result_state.get("mcp_context", {})
        messages = result_state.get("messages", [])
        response_text = messages[0].content if messages else ""
        return {
            "agent": agent_name,
            "data": mcp_data.get(agent_name, mcp_data),
            "response": response_text,
        }
