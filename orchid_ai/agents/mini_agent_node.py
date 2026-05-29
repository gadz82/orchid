"""Mini-agent runtime node.

A single LangGraph node, registered once per opt-in parent under the
name ``f"{parent_name}_mini"``.  The graph's fork router fans out one
``Send`` per sub-task; LangGraph runs every Send branch in parallel
through this same node and joins them at the aggregator.

Each invocation:

1. Reads its sub-task identity from the ``_active_mini_*`` sentinel
   keys on state.
2. Renders MCP capabilities against the parent's MCP clients.
3. Filters the parent's tool inventory to ``allowed_tools``.
4. Builds a focused system prompt (``parent.prompt`` + sub-task
   instruction + tools list).
5. Runs an ``AgenticLoop`` (with ``is_mini=True`` and the resolved
   ``tool_subset``) wrapped in ``asyncio.wait_for``.
6. Emits a ``MiniAgentOutcome`` (status: ``ok`` / ``failed`` /
   ``timeout``) into ``mini_agent_outcomes[f"{parent}#{mini_id}"]``.

The mini node never re-runs the decomposer — it bypasses
``GenericAgent.run()`` entirely, which is the structural guarantee
behind the "mini-of-a-mini" guard.

Single-responsibility: this module owns ONE node — the mini agent.
The decomposer lives in ``mini_agent_decomposer.py``; the aggregator
in ``mini_agent_aggregator.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..config.schema import OrchidAgentConfig
from ..core.mcp import OrchidMCPClient
from ..core.state import OrchidAuthContext

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


__all__ = [
    "MiniAgentOutcome",
    "MiniAgentRuntimeError",
    "mini_agent_node_factory",
]


# LangGraph signals graph-control flow (``interrupt`` for HITL,
# ``Command`` re-routing) by raising subclasses of
# ``GraphBubbleUp``.  These MUST escape the mini node's broad
# ``except Exception`` block — catching them would convert a
# legitimate suspension request into a ``status="failed"`` outcome.
try:  # pragma: no cover — exercised by tests when langgraph is installed.
    from langgraph.errors import GraphBubbleUp as _GraphBubbleUp

    _GRAPH_BUBBLE_UP_EXCS: tuple[type[BaseException], ...] = (_GraphBubbleUp,)
except ImportError:  # pragma: no cover — fallback when langgraph is not installed.
    _GRAPH_BUBBLE_UP_EXCS = ()


# ── Outcome model ──────────────────────────────────────────────


class MiniAgentOutcome(BaseModel):
    """One mini-agent's outcome — written to ``mini_agent_outcomes``.

    The aggregator filters by parent prefix and synthesises the
    parent's final answer from the collection.
    """

    mini_id: str
    sub_task_description: str
    status: Literal["ok", "failed", "timeout"]
    summary: str | None = None
    error: str | None = None
    duration_ms: int = 0
    tool_results: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class MiniAgentRuntimeError(Exception):
    """Raised by the mini-agent node when its sub-task payload is
    structurally invalid (e.g. missing sentinel keys).  This is a
    framework-level invariant violation, distinct from sub-task
    execution failures (which are recorded as ``status=failed``
    outcomes).
    """


# ── Factory ────────────────────────────────────────────────────


def mini_agent_node_factory(
    *,
    parent_config: OrchidAgentConfig,
    chat_model: BaseChatModel,
    mcp_clients: list[OrchidMCPClient],
) -> Callable[[dict[str, Any]], Any]:
    """Build a LangGraph node function that runs a single mini-agent.

    The closure captures the parent's chat model and MCP client
    instances — the spec requires "the same ``OrchidAuthContext`` and
    ``OrchidMCPClient`` instances as the parent (no per-mini auth or
    MCP-session creation)" so MCP capability caches stay warm and
    OAuth tokens are not re-resolved per fork.
    """
    parent_name = parent_config.name
    timeout = parent_config.mini_agent.timeout_seconds

    async def mini_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        sub_task_payload = state.get("_active_mini_subtask") or {}
        mini_id = state.get("_active_mini_id") or sub_task_payload.get("id") or "mini_unknown"
        active_parent = state.get("_active_mini_parent") or parent_name
        if active_parent != parent_name:
            # Defence-in-depth: if a Send mis-routes a sub-task across
            # parents, refuse to run rather than corrupt the wrong
            # ``mini_agent_outcomes`` slot.
            raise MiniAgentRuntimeError(
                f"mini node for parent '{parent_name}' invoked with _active_mini_parent='{active_parent}'",
            )
        slot_key = f"{parent_name}#{mini_id}"
        description = sub_task_payload.get("description", mini_id)
        instruction = sub_task_payload.get("instruction", "")
        tool_subset: list[str] = state.get("_active_mini_tool_subset") or list(
            sub_task_payload.get("allowed_tools") or [],
        )

        auth: OrchidAuthContext | None = state.get("auth_context")
        if auth is None:
            return _emit_outcome(
                parent_name=parent_name,
                mini_id=mini_id,
                description=description,
                status="failed",
                error="missing auth_context on state",
                duration_ms=_elapsed_ms(start),
            )

        try:
            outcome_dict = await _run_mini_with_timeout(
                parent_config=parent_config,
                chat_model=chat_model,
                mcp_clients=mcp_clients,
                auth=auth,
                state=state,
                mini_id=mini_id,
                description=description,
                instruction=instruction,
                tool_subset=tool_subset,
                timeout=timeout,
            )
            outcome_dict["duration_ms"] = _elapsed_ms(start)
            return _wrap_state_update(parent_name, slot_key, outcome_dict)
        except _GRAPH_BUBBLE_UP_EXCS:
            # HITL ``interrupt()`` raises ``GraphInterrupt`` (a
            # subclass of ``GraphBubbleUp``) which LangGraph's
            # runtime catches at the node boundary to pause execution
            # and surface the approval prompt.  We MUST NOT swallow
            # it — let it propagate so the graph suspends and the
            # frontend can collect the approval before resuming.  The
            # checkpointer + thread_id machinery picks the same mini
            # back up via ``Command(resume=...)``.
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "[%s/%s] mini-agent exceeded timeout of %ds",
                parent_name,
                mini_id,
                timeout,
            )
            return _emit_outcome(
                parent_name=parent_name,
                mini_id=mini_id,
                description=description,
                status="timeout",
                error=f"timed out after {timeout}s",
                duration_ms=_elapsed_ms(start),
                slot_key=slot_key,
            )
        except Exception as exc:  # noqa: BLE001 — fault isolation
            logger.error(
                "[%s/%s] mini-agent raised %s: %s",
                parent_name,
                mini_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return _emit_outcome(
                parent_name=parent_name,
                mini_id=mini_id,
                description=description,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(start),
                slot_key=slot_key,
            )

    mini_agent_node.__name__ = f"{parent_name}_mini"
    return mini_agent_node


# ── Inner runner — separated so it can be timed-out cleanly ──


async def _run_mini_with_timeout(
    *,
    parent_config: OrchidAgentConfig,
    chat_model: BaseChatModel,
    mcp_clients: list[OrchidMCPClient],
    auth: OrchidAuthContext,
    state: dict[str, Any],
    mini_id: str,
    description: str,
    instruction: str,
    tool_subset: list[str],
    timeout: int,
) -> dict[str, Any]:
    """Execute the inner agentic loop with the configured timeout.

    Returns a dict suitable for ``MiniAgentOutcome(**...)`` — the
    caller wraps it with the slot key, durations, and state delta.
    """

    async def _inner() -> dict[str, Any]:
        return await _run_inner_loop(
            parent_config=parent_config,
            chat_model=chat_model,
            mcp_clients=mcp_clients,
            auth=auth,
            state=state,
            mini_id=mini_id,
            description=description,
            instruction=instruction,
            tool_subset=tool_subset,
        )

    return await asyncio.wait_for(_inner(), timeout=timeout)


async def _run_inner_loop(
    *,
    parent_config: OrchidAgentConfig,
    chat_model: BaseChatModel,
    mcp_clients: list[OrchidMCPClient],
    auth: OrchidAuthContext,
    state: dict[str, Any],
    mini_id: str,
    description: str,
    instruction: str,
    tool_subset: list[str],
) -> dict[str, Any]:
    """Build the AgenticLoop inputs and run it.  Returns an outcome dict."""
    # Local import to keep the module-level dependency surface tight.
    from .agentic_loop import AgenticLoop
    from .generic_agent import GenericAgent
    from .mcp_dispatcher import MCPDispatcher
    from .tools import build_langchain_tools

    # ── Render parent's MCP capabilities (per-process cache hits) ──
    dispatcher = MCPDispatcher(mcp_clients=mcp_clients, server_configs=parent_config.mcp_servers)
    caps = await dispatcher.render_capabilities(auth, agent_name=f"{parent_config.name}.{mini_id}")

    # ── Build the parent's full tool inventory and filter to the subset ──
    builtin_tool_names, builtin_tool_defs = _builtin_tools_for_parent(parent_config)
    mcp_tool_defs = MCPDispatcher.mcp_tools_to_litellm(
        [t for t in caps.raw_tools if t["name"] not in builtin_tool_names],
    )

    allowed = set(tool_subset) if tool_subset else None
    if allowed is not None:
        builtin_tool_defs = [td for td in builtin_tool_defs if td["function"]["name"] in allowed]
        mcp_tool_defs = [td for td in mcp_tool_defs if td["function"]["name"] in allowed]

    all_tool_defs = mcp_tool_defs + builtin_tool_defs

    lc_tools = build_langchain_tools(
        builtin_names=builtin_tool_names if allowed is None else builtin_tool_names & allowed,
        builtin_tool_defs=builtin_tool_defs,
        mcp_tool_defs=mcp_tool_defs,
        mcp_tool_client_map=caps.tool_client_map,
        auth=auth,
        agent_name=f"{parent_config.name}.{mini_id}",
        approval_tools=parent_config.approval_tools or None,
    )
    tool_map: dict[str, Any] = {t.name: t for t in lc_tools}

    # ── Resolve parallel-safety inheritance from the parent ──
    parallel_safety = _inherit_parallel_safety(
        parent_config=parent_config,
        tool_map=tool_map,
        builtin_tool_names=builtin_tool_names,
        caps=caps,
    )

    # ── Build the focused system prompt ──
    system_prompt = _build_mini_system_prompt(
        parent_prompt=parent_config.prompt,
        instruction=instruction,
        tool_subset=list(tool_map.keys()),
        caps_descriptions=_tool_descriptions(caps, builtin_tool_defs),
        template=parent_config.mini_agent.system_prompt_template,
    )

    # ── Build messages: system + parent's conversation history + user query ──
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(
        GenericAgent.extract_conversation_history(
            state,
            max_turns=20,
            max_chars=1000,
        )
    )
    user_query = GenericAgent.extract_user_query(state)
    if user_query:
        messages.append({"role": "user", "content": user_query})

    # ── Run the agentic loop ──
    llm_config = parent_config.llm
    loop = AgenticLoop(
        agent_name=f"{parent_config.name}.{mini_id}",
        chat_model=chat_model,
        tool_map=tool_map,
        all_tool_defs=all_tool_defs,
        temperature=llm_config.temperature if llm_config else 0.2,
        parallel_safety=parallel_safety,
        tool_subset=list(tool_map.keys()) if tool_subset else None,
        is_mini=True,
        max_tool_rounds=parent_config.max_tool_rounds,
        max_consecutive_dupes=parent_config.max_consecutive_dupes,
    )

    final_text, tool_results = await loop.run(messages)
    summary = final_text or _fallback_summary(tool_results)
    return {
        "mini_id": mini_id,
        "sub_task_description": description,
        "status": "ok",
        "summary": summary,
        "error": None,
        "tool_results": _stringify_tool_results(tool_results),
    }


# ── Helpers ────────────────────────────────────────────────────


def _builtin_tools_for_parent(
    parent_config: OrchidAgentConfig,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Mirror ``GenericAgent._builtin_tools_to_litellm`` against parent_config."""
    from .tool_utils import tools_to_litellm_format

    return tools_to_litellm_format(parent_config.tools)


def _inherit_parallel_safety(
    *,
    parent_config: OrchidAgentConfig,
    tool_map: dict[str, Any],
    builtin_tool_names: set[str],
    caps: Any,  # MCPCapabilities — typed loosely to avoid circular import
) -> dict[str, bool] | None:
    """Resolve per-tool parallel-safety using the parent's settings.

    Honours ``parent_config.parallel_tools`` so minis inherit the parent's
    parallel-safety configuration.  Mirrors the precedence rules in
    ``GenericAgent._resolve_parallel_safety`` against the (already
    filtered) mini ``tool_map``.
    """
    from .tool_utils import resolve_parallel_safety

    mcp_overrides: dict[str, bool] = {}
    for server in parent_config.mcp_servers:
        for tool in server.tools:
            if tool.parallel_safe is not None:
                mcp_overrides[tool.name] = tool.parallel_safe

    return resolve_parallel_safety(
        tool_map=tool_map,
        builtin_tool_names=builtin_tool_names,
        caps=caps,
        parallel_tools_enabled=bool(parent_config.parallel_tools),
        approval_tools=parent_config.approval_tools,
        parallel_safe_builtin_tools=parent_config.parallel_safe_builtin_tools,
        mcp_parallel_overrides=mcp_overrides,
    )


def _build_mini_system_prompt(
    *,
    parent_prompt: str,
    instruction: str,
    tool_subset: list[str],
    caps_descriptions: dict[str, str],
    template: str | None = None,
) -> str:
    """Compose the mini's focused system prompt.

    When ``template`` is supplied it is resolved via
    :py:meth:`str.format` with the placeholders ``{parent_prompt}``,
    ``{instruction}``, and ``{tool_list}`` — the last being a newline-
    joined ``- name: description`` bullet list, or the empty string
    when the mini has no tools.  ``None`` (default) uses the built-in
    assembly below.
    """
    tool_list = ""
    if tool_subset:
        bullets = [f"- {name}: {caps_descriptions.get(name, name)}" for name in tool_subset]
        tool_list = "\n".join(bullets)

    if template is not None:
        return template.format(
            parent_prompt=parent_prompt,
            instruction=instruction,
            tool_list=tool_list,
        )

    parts = [parent_prompt]
    if instruction:
        parts.append("\n\n" + instruction)
    if tool_list:
        parts.append("\n\nTools available to you:\n" + tool_list)
    return "".join(parts)


def _tool_descriptions(
    caps: Any,
    builtin_tool_defs: list[dict[str, Any]],
) -> dict[str, str]:
    """One-line description per tool name (MCP + built-in)."""
    out: dict[str, str] = {}
    for raw in getattr(caps, "raw_tools", []) or []:
        name = raw.get("name")
        if name:
            out[name] = (raw.get("description") or "").strip().splitlines()[0:1] or [""]
            out[name] = out[name][0]
    for td in builtin_tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name")
        if name:
            out[name] = (fn.get("description") or "").strip().splitlines()[0:1] or [""]
            out[name] = out[name][0]
    return out


def _stringify_tool_results(tool_results: dict[str, Any]) -> dict[str, str]:
    """Flatten tool_results into a string-only dict so the outcome is
    cleanly serialisable into LangGraph state.  Non-string values are
    JSON-encoded; ``None`` becomes the empty string.
    """
    out: dict[str, str] = {}
    for k, v in (tool_results or {}).items():
        if v is None:
            out[k] = ""
        elif isinstance(v, str):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, default=str)
            except Exception:
                out[k] = str(v)
    return out


def _fallback_summary(tool_results: dict[str, Any]) -> str:
    """When the loop exhausted rounds without producing text, hand the
    aggregator a JSON dump of the tool results so it has something to
    work with.  Capped to a reasonable size.
    """
    if not tool_results:
        return ""
    try:
        return json.dumps(tool_results, default=str)[:4000]
    except Exception:
        return ""


def _wrap_state_update(parent_name: str, slot_key: str, outcome: dict[str, Any]) -> dict[str, Any]:
    """Shape a state update that writes the outcome plus the per-mini
    shadow ``mcp_context`` slot for trace inspection.

    Also emits the two ``mini_agent.{started,finished}`` lifecycle
    events into the ``messages`` channel so the streaming router can
    surface them as SSE events.  Both events ride the same state
    delta because the mini node is a single LangGraph node — they
    arrive together at the message stream but the SSE consumer can
    still distinguish + time-order them by their ``type`` field.
    """
    from ..observability import make_event_message

    tool_results = outcome.get("tool_results") or {}
    started_event = make_event_message(
        "mini_agent.started",
        {
            "parent": parent_name,
            "mini_id": outcome.get("mini_id", ""),
            "description": outcome.get("sub_task_description", ""),
        },
    )
    finished_payload: dict[str, Any] = {
        "parent": parent_name,
        "mini_id": outcome.get("mini_id", ""),
        "status": outcome.get("status", ""),
        "duration_ms": outcome.get("duration_ms", 0),
    }
    if outcome.get("error"):
        finished_payload["error"] = outcome["error"]
    finished_event = make_event_message("mini_agent.finished", finished_payload)
    return {
        "messages": [started_event, finished_event],
        "mini_agent_outcomes": {slot_key: outcome},
        "mcp_context": {slot_key: {"tool_results": tool_results}},
    }


def _emit_outcome(
    *,
    parent_name: str,
    mini_id: str,
    description: str,
    status: Literal["ok", "failed", "timeout"],
    error: str | None = None,
    duration_ms: int = 0,
    slot_key: str | None = None,
) -> dict[str, Any]:
    """Build a state update for a failed/timed-out outcome (no tool_results)."""
    outcome = {
        "mini_id": mini_id,
        "sub_task_description": description,
        "status": status,
        "summary": None,
        "error": error,
        "duration_ms": duration_ms,
        "tool_results": {},
    }
    return _wrap_state_update(parent_name, slot_key or f"{parent_name}#{mini_id}", outcome)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
