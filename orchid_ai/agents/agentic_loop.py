"""
Agentic tool-calling loop — extracted from GenericAgent (SRP).

Manages the multi-round LLM → tool-call → result cycle with:
- Unified MCP + built-in tool dispatch via LangChain tool wrappers
- Duplicate call detection and consecutive-dupe stripping
- Max-round safety (prevents infinite loops)
- HITL interrupt for tools requiring approval
- Per-call error handling
- Optional parallel dispatch of read-only / idempotent tools within a
  single round (Phase A) — opt-in via ``parallel_safety`` map.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any


logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")

_MAX_TOOL_ROUNDS = 15
_MAX_CONSECUTIVE_DUPES = 2


def _is_parallel_safe(
    tool_name: str,
    parallel_safety: dict[str, bool] | None,
) -> bool:
    """Resolve whether ``tool_name`` may join the parallel batch.

    Pure lookup against a pre-resolved map.  ``GenericAgent`` builds the
    map upfront with the full precedence chain (explicit YAML override
    > MCP ``readOnlyHint`` annotation > ``False``) and forces
    ``False`` for any tool whose ``requires_approval`` is set.  A
    ``None`` map means the agent has not opted in to ``parallel_tools``
    at all — the loop runs strictly sequentially.
    """
    if parallel_safety is None:
        return False
    return bool(parallel_safety.get(tool_name, False))


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
        parallel_safety: dict[str, bool] | None = None,
        tool_subset: list[str] | None = None,
        is_mini: bool = False,
    ) -> None:
        self._agent_name = agent_name
        self._chat_model = chat_model
        self._temperature = temperature
        # ``is_mini`` is metadata used for log labelling and to mark
        # this loop as belonging to a forked sub-task.  It does not
        # gate any behaviour inside the loop — duplicate-detection
        # state is per-instance so cross-mini contamination is not
        # possible by construction.
        self._is_mini = is_mini

        # ``tool_subset`` filters the tools the LLM can see.  Passed
        # by ``mini_agent_node`` to enforce the decomposer's
        # ``allowed_tools`` allowlist; the parent agent runs with
        # ``tool_subset=None`` to expose its full inventory.  The
        # filter applies BEFORE ``bind_tools`` and updates ``tool_map``
        # to match — defence-in-depth so the LLM can neither see nor
        # call tools outside the subset.
        if tool_subset is not None:
            allowed = set(tool_subset)
            self._tool_map = {n: t for n, t in tool_map.items() if n in allowed}
            self._all_tool_defs = [td for td in all_tool_defs if td.get("function", {}).get("name") in allowed]
        else:
            self._tool_map = tool_map
            self._all_tool_defs = all_tool_defs

        # When ``None``, the loop runs strictly sequentially (today's
        # behaviour).  When provided, every key whose value is ``True``
        # is eligible to join the parallel batch within a round.  See
        # ``_dispatch_tool_calls`` for the partition rules.
        self._parallel_safety = parallel_safety

        # Loop state
        self._seen_calls: dict[str, str] = {}
        self._consecutive_dupes = 0
        self._tool_results: dict[str, Any] = {}

    @property
    def is_mini(self) -> bool:
        """True when this loop runs inside a forked mini-agent."""
        return self._is_mini

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
        """Execute each tool call with duplicate detection and HITL.

        Two execution modes share the same observable outcome — every
        ``ToolMessage`` is appended to ``messages`` in the original
        ``tool_calls`` order:

        - **Sequential** (default, used when ``parallel_safety`` is
          ``None`` OR when any call in the round requires approval):
          identical to today's implementation — calls run one at a
          time, duplicate detection and HITL interrupts fire at their
          original positions.
        - **Mixed** (used when at least one call is parallel-safe and
          no call requires approval): the round is partitioned into a
          ``parallel_batch`` (gathered via ``asyncio.gather``) and a
          ``sequential_tail`` (everything else, in original order).
          The parallel batch runs first, then the sequential tail —
          the spec contract is "approval serialises, safe ones gather,
          tail order preserved".
        """
        any_requires_approval = any(self._requires_approval(tc) for tc in tool_calls)
        if self._parallel_safety is None or any_requires_approval:
            await self._dispatch_sequential(tool_calls, messages, round_num)
            return

        await self._dispatch_mixed(tool_calls, messages, round_num)

    async def _dispatch_sequential(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list,
        round_num: int,
    ) -> None:
        """Run every tool call serially — preserves today's behaviour."""
        from langchain_core.messages import ToolMessage

        for tc in tool_calls:
            fn_name, fn_args, tc_id, call_key = self._unpack(tc)
            logger.info(
                "[%s] Tool call #%d -> %s | args: %s",
                self._agent_name,
                round_num + 1,
                fn_name,
                fn_args,
            )
            result_text = await self._execute_one(fn_name, fn_args, call_key, round_num)
            self._tool_results[fn_name] = result_text
            messages.append(ToolMessage(content=result_text, tool_call_id=tc_id))

    async def _dispatch_mixed(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list,
        round_num: int,
    ) -> None:
        """Run parallel-safe calls in a gather, then the rest sequentially.

        ``ToolMessage``s are appended to ``messages`` in the original
        ``tool_calls`` order regardless of which bucket each call lands
        in — callers (and the LLM on the next round) see the round as
        if it had run serially.
        """
        from langchain_core.messages import ToolMessage

        parallel_indices: list[int] = []
        sequential_indices: list[int] = []
        unpacked: list[tuple[str, dict[str, Any], str, str]] = []
        for idx, tc in enumerate(tool_calls):
            fn_name, fn_args, tc_id, call_key = self._unpack(tc)
            unpacked.append((fn_name, fn_args, tc_id, call_key))
            if self._is_eligible_for_parallel(fn_name, call_key):
                parallel_indices.append(idx)
            else:
                sequential_indices.append(idx)

        results: list[str | None] = [None] * len(tool_calls)

        # ── Parallel batch ─────────────────────────────────
        if parallel_indices:
            logger.info(
                "[%s] Round #%d parallel batch (%d tool(s)): %s",
                self._agent_name,
                round_num + 1,
                len(parallel_indices),
                [unpacked[i][0] for i in parallel_indices],
            )
            gather_start = time.perf_counter()
            coros = [self._execute_parallel_leg(*unpacked[i]) for i in parallel_indices]
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            gather_elapsed = (time.perf_counter() - gather_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s][loop] round=%d parallel_gather n=%d took %.1f ms",
                self._agent_name,
                round_num + 1,
                len(parallel_indices),
                gather_elapsed,
            )
            for slot, idx in enumerate(parallel_indices):
                fn_name, _, _, _ = unpacked[idx]
                outcome = gathered[slot]
                if isinstance(outcome, BaseException):
                    text = f"[Tool error] {outcome!r}"
                    logger.error(
                        "[%s] Parallel tool '%s' raised %s: %s",
                        self._agent_name,
                        fn_name,
                        type(outcome).__name__,
                        outcome,
                    )
                else:
                    text = outcome
                results[idx] = text
                self._tool_results[fn_name] = text

        # ── Sequential tail ────────────────────────────────
        for idx in sequential_indices:
            fn_name, fn_args, _, call_key = unpacked[idx]
            logger.info(
                "[%s] Tool call #%d -> %s | args: %s (sequential)",
                self._agent_name,
                round_num + 1,
                fn_name,
                fn_args,
            )
            text = await self._execute_one(fn_name, fn_args, call_key, round_num)
            results[idx] = text
            self._tool_results[fn_name] = text

        # ── Append in original order ───────────────────────
        for idx, tc in enumerate(tool_calls):
            tc_id = unpacked[idx][2]
            messages.append(ToolMessage(content=results[idx] or "", tool_call_id=tc_id))

    # ── Dispatch helpers ───────────────────────────────────────

    def _unpack(self, tc: dict[str, Any]) -> tuple[str, dict[str, Any], str, str]:
        """Decompose one ``tool_call`` payload into its useful pieces."""
        fn_name = tc["name"]
        fn_args = tc.get("args", {})
        tc_id = tc.get("id", "")
        call_key = f"{fn_name}|{json.dumps(fn_args, sort_keys=True)}"
        return fn_name, fn_args, tc_id, call_key

    def _requires_approval(self, tc: dict[str, Any]) -> bool:
        """True when the tool referenced by ``tc`` is HITL-gated."""
        tool = self._tool_map.get(tc.get("name", ""))
        return bool(getattr(tool, "requires_approval", False))

    def _is_eligible_for_parallel(self, fn_name: str, call_key: str) -> bool:
        """True iff this call may join the round's parallel batch.

        Eligibility: the agent opted in (``parallel_safety`` provided),
        the tool resolves to ``True`` in the safety map, the tool is
        known, the call is NOT a duplicate (duplicate handling depends
        on shared mutable state and stays sequential), and the tool is
        NOT HITL-gated (defence-in-depth — the parent already filters
        whole rounds containing approvals, but we belt-and-brace here).
        """
        if not _is_parallel_safe(fn_name, self._parallel_safety):
            return False
        if fn_name not in self._tool_map:
            return False
        if call_key in self._seen_calls:
            return False
        tool = self._tool_map[fn_name]
        if getattr(tool, "requires_approval", False):
            return False
        return True

    async def _execute_one(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        call_key: str,
        round_num: int,
    ) -> str:
        """Single-call dispatch with duplicate detection — sequential path."""
        if call_key in self._seen_calls:
            return self._handle_duplicate(call_key, fn_name, round_num)
        if fn_name in self._tool_map:
            return await self._dispatch_single(fn_name, fn_args, call_key)
        logger.error("[%s] Unknown tool '%s' in agentic loop", self._agent_name, fn_name)
        return f"[Error] Unknown tool '{fn_name}'"

    async def _execute_parallel_leg(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        _tc_id: str,
        call_key: str,
    ) -> str:
        """Single-call dispatch for a leg inside ``asyncio.gather``.

        Eligibility (``_is_eligible_for_parallel``) already excluded
        duplicates, unknowns, and HITL — so this is a straight call to
        the underlying tool wrapper.  Errors raised by the wrapper
        propagate; ``asyncio.gather(return_exceptions=True)`` catches
        them upstream and the caller renders them as ``[Tool error]``
        ``ToolMessage``s.
        """
        return await self._dispatch_single(fn_name, fn_args, call_key)

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
