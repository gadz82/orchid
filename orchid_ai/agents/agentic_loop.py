"""
Agentic tool-calling loop — extracted from GenericAgent (SRP).

Manages the multi-round LLM → tool-call → result cycle with:
- Unified MCP + built-in tool dispatch via LangChain tool wrappers
- Duplicate call detection and consecutive-dupe stripping
- Max-round safety (prevents infinite loops)
- HITL interrupt for tools requiring approval
- Per-call error handling
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any


logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")

_MAX_TOOL_ROUNDS = 15
_MAX_CONSECUTIVE_DUPES = 2


class AgenticLoop:
    """Runs the multi-round tool-calling loop for a single agent invocation.

    Created per-invocation by ``GenericAgent._agentic_tool_loop()``.
    Owns the loop state (duplicate tracking, round counter) and delegates
    tool dispatch to pre-built LangChain ``BaseTool`` wrappers.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        chat_model: Any,
        tool_map: dict[str, Any],
        all_tool_defs: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> None:
        self._agent_name = agent_name
        self._chat_model = chat_model
        self._tool_map = tool_map
        self._all_tool_defs = all_tool_defs
        self._temperature = temperature

        # Loop state
        self._seen_calls: dict[str, str] = {}
        self._consecutive_dupes = 0
        self._tool_results: dict[str, Any] = {}

    @property
    def tool_results(self) -> dict[str, Any]:
        return self._tool_results

    async def run(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, dict[str, Any]]:
        """Execute the tool-calling loop.

        Returns ``(final_text, tool_results)``.  ``final_text`` is ``None``
        when the loop exhausted rounds without the LLM producing a text
        response (caller should fall back to summarisation).
        """
        loop_start = time.perf_counter()
        model_with_tools = self._chat_model.bind_tools(self._all_tool_defs)
        perf_logger.info(
            "[PERF][agent=%s][loop] start tools_available=%d",
            self._agent_name,
            len(self._all_tool_defs),
        )

        for round_num in range(_MAX_TOOL_ROUNDS):
            active_model = self._pick_model(model_with_tools, round_num)

            llm_start = time.perf_counter()
            ai_msg = await self._invoke_llm(active_model, messages, round_num)
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            if isinstance(ai_msg, str):
                # Error message — return early
                perf_logger.warning(
                    "[PERF][agent=%s][loop] round=%d LLM error after %.1f ms",
                    self._agent_name,
                    round_num + 1,
                    llm_elapsed,
                )
                return ai_msg, self._tool_results

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            perf_logger.info(
                "[PERF][agent=%s][loop] round=%d LLM call took %.1f ms (tool_calls=%d)",
                self._agent_name,
                round_num + 1,
                llm_elapsed,
                len(tool_calls),
            )

            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                final_text = ai_msg.content or ""
                total_elapsed = (time.perf_counter() - loop_start) * 1000
                perf_logger.info(
                    "[PERF][agent=%s][loop] DONE rounds=%d total=%.1f ms (final_text=%d chars)",
                    self._agent_name,
                    round_num + 1,
                    total_elapsed,
                    len(final_text),
                )
                logger.info("[%s] LLM responded after %d tool round(s)", self._agent_name, round_num)
                return final_text, self._tool_results

            dispatch_start = time.perf_counter()
            await self._dispatch_tool_calls(ai_msg.tool_calls, messages, round_num)
            dispatch_elapsed = (time.perf_counter() - dispatch_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s][loop] round=%d tool dispatch took %.1f ms",
                self._agent_name,
                round_num + 1,
                dispatch_elapsed,
            )

        total_elapsed = (time.perf_counter() - loop_start) * 1000
        perf_logger.warning(
            "[PERF][agent=%s][loop] HIT_MAX_ROUNDS rounds=%d total=%.1f ms",
            self._agent_name,
            _MAX_TOOL_ROUNDS,
            total_elapsed,
        )
        logger.warning("[%s] Hit max tool rounds (%d)", self._agent_name, _MAX_TOOL_ROUNDS)
        return None, self._tool_results

    # ── Internal helpers ───────────────────────────────────────

    def _pick_model(self, model_with_tools: Any, round_num: int) -> Any:
        """Strip tools after consecutive duplicate calls to force text output."""
        if self._consecutive_dupes >= _MAX_CONSECUTIVE_DUPES:
            logger.warning(
                "[%s] %d consecutive duplicate calls — forcing text-only response",
                self._agent_name,
                self._consecutive_dupes,
            )
            return self._chat_model
        return model_with_tools

    async def _invoke_llm(
        self,
        model: Any,
        messages: list,
        round_num: int,
    ) -> Any:
        """Call the LLM with error handling. Returns AIMessage or error string."""
        try:
            return await model.ainvoke(messages, temperature=self._temperature)
        except Exception as exc:
            from ..llm_errors import format_llm_error

            return format_llm_error(exc, context=f"{self._agent_name} round {round_num}")

    async def _dispatch_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list,
        round_num: int,
    ) -> None:
        """Execute each tool call with duplicate detection and HITL."""
        from langchain_core.messages import ToolMessage

        for tc in tool_calls:
            fn_name = tc["name"]
            fn_args = tc.get("args", {})
            tc_id = tc.get("id", "")

            logger.info("[%s] Tool call #%d -> %s | args: %s", self._agent_name, round_num + 1, fn_name, fn_args)

            call_key = f"{fn_name}|{json.dumps(fn_args, sort_keys=True)}"

            if call_key in self._seen_calls:
                result_text = self._handle_duplicate(call_key, fn_name, round_num)
            elif fn_name in self._tool_map:
                result_text = await self._dispatch_single(fn_name, fn_args, call_key)
            else:
                result_text = f"[Error] Unknown tool '{fn_name}'"
                logger.error("[%s] Unknown tool '%s' in agentic loop", self._agent_name, fn_name)

            self._tool_results[fn_name] = result_text
            messages.append(ToolMessage(content=result_text, tool_call_id=tc_id))

    def _handle_duplicate(self, call_key: str, fn_name: str, round_num: int) -> str:
        self._consecutive_dupes += 1
        logger.warning(
            "[%s] Duplicate tool call #%d -> %s (%d consecutive)",
            self._agent_name,
            round_num + 1,
            fn_name,
            self._consecutive_dupes,
        )
        return (
            "You already called this tool with the same parameters. "
            "Here is the previous result (do NOT call it again — "
            "summarise this data for the user instead):\n\n" + self._seen_calls[call_key]
        )

    async def _dispatch_single(self, fn_name: str, fn_args: dict, call_key: str) -> str:
        """Dispatch a single tool call, handling HITL approval."""
        self._consecutive_dupes = 0
        tool = self._tool_map[fn_name]
        tool_kind = type(tool).__name__

        if tool.requires_approval:
            from langgraph.types import interrupt

            logger.info("[%s] Tool '%s' requires approval — interrupting", self._agent_name, fn_name)
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "tool": fn_name,
                    "args": fn_args,
                    "agent": self._agent_name,
                }
            )
            if not decision.get("approved"):
                return "Tool execution cancelled by user."

        invoke_start = time.perf_counter()
        result_text = await tool.ainvoke(fn_args)
        invoke_elapsed = (time.perf_counter() - invoke_start) * 1000
        is_error = result_text.startswith("[Tool error]")
        perf_logger.info(
            "[PERF][agent=%s][tool] %s name=%s took %.1f ms (out_chars=%d, error=%s)",
            self._agent_name,
            tool_kind,
            fn_name,
            invoke_elapsed,
            len(result_text),
            is_error,
        )
        if not is_error:
            self._seen_calls[call_key] = result_text
        return result_text
