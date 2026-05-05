"""Mini-agent aggregator node.

Runs once per parent agent per turn after every mini in that turn
has produced a final outcome.  LangGraph's join semantics ensure all
parallel ``Send``s have completed before this node fires.

Two paths:

1. **All-failed short-circuit** — if zero outcomes succeeded, emit
   a deterministic error ``AIMessage`` and skip the LLM call.  No
   need to ask the synthesis model to "frame" something with no
   data.
2. **Synthesis** — run an LLM call against the parent's chat model
   with the default aggregator prompt (or the parent's override).
   Cite each outcome's description; never invent content for
   failed/timed-out sub-tasks.

The aggregator writes:

- ``messages`` — one synthesised ``AIMessage`` named after the
  parent, the supervisor's only visible artefact for this turn.
- ``mcp_context[parent]`` — merged tool_results from successful
  outcomes only, plus the synthesised text and the full outcome
  list (for trace inspection).

Single-responsibility: synthesis only — decomposition lives in
``mini_agent_decomposer.py``, mini execution in ``mini_agent_node.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..config.schema import OrchidAgentConfig

logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_AGGREGATOR_PROMPT",
    "aggregator_node_factory",
]


DEFAULT_AGGREGATOR_PROMPT = """\
You are synthesising the final answer for the "{agent_name}" agent.
The original user request was: {user_query}

You ran {n} independent sub-tasks in parallel. Their outcomes:
{outcome_block}

Produce ONE coherent answer for the user. Rules:
  - If failures or timeouts are blocking, say so explicitly. Tell the user
    which sub-tasks succeeded and which didn't.
  - Do NOT invent results for failed/timed-out sub-tasks.
  - Cite the sub-task description when referencing its findings.
  - Do not mention "mini-agents" or internal architecture — speak as the agent.
"""


def aggregator_node_factory(
    *,
    parent_config: OrchidAgentConfig,
    chat_model: BaseChatModel,
) -> Callable[[dict[str, Any]], Any]:
    """Build a LangGraph node that synthesises a parent's mini outcomes.

    The closure captures the parent's chat model and config; it does
    NOT capture per-request state, so it's safe to share across all
    invocations of this parent.
    """
    parent_name = parent_config.name
    prompt_template = parent_config.mini_agent.aggregator_prompt or DEFAULT_AGGREGATOR_PROMPT

    async def aggregator_node(state: dict[str, Any]) -> dict[str, Any]:
        outcomes = _collect_outcomes(state, parent_name)
        if not outcomes:
            # The graph never reaches the aggregator without outcomes,
            # but be defensive: emit a short-circuit error so a stuck
            # turn does not leave the supervisor waiting forever.
            return _short_circuit_message(
                parent_name=parent_name,
                outcomes=outcomes,
                reason="no mini-agent outcomes were recorded",
            )

        # ── All-failed short-circuit (spec §11.1, §12) ────
        successful = [o for o in outcomes if o.get("status") == "ok"]
        if not successful:
            error_summary = _summarise_failures(outcomes)
            return _short_circuit_message(
                parent_name=parent_name,
                outcomes=outcomes,
                reason=error_summary,
            )

        # ── LLM-framed synthesis ───────────────────────────
        from ..core.agent import OrchidAgent

        user_query = OrchidAgent.extract_user_query(state)
        prompt = prompt_template.format(
            agent_name=parent_name,
            user_query=user_query or "(no query found)",
            n=len(outcomes),
            outcome_block=_render_outcome_block(outcomes),
        )

        try:
            response = await chat_model.ainvoke([{"role": "system", "content": prompt}])
            synthesis = getattr(response, "content", "") or ""
        except Exception as exc:  # noqa: BLE001 — fault isolation
            logger.error(
                "[%s/aggregator] synthesis LLM call failed: %s",
                parent_name,
                exc,
                exc_info=True,
            )
            synthesis = _fallback_synthesis(outcomes)

        merged_tool_results = _merge_successful_tool_results(outcomes)
        from ..observability import make_event_message

        aggregated_event = make_event_message(
            "mini_agent.aggregated",
            {
                "parent": parent_name,
                "outcomes": [{"mini_id": o.get("mini_id", ""), "status": o.get("status", "")} for o in outcomes],
            },
        )
        return {
            # Emit the lifecycle event BEFORE the user-visible AIMessage so
            # the streaming router translates ``mini_agent.aggregated`` to
            # SSE before the synthesised tokens land.
            "messages": [aggregated_event, AIMessage(content=synthesis, name=parent_name)],
            "mcp_context": {
                parent_name: {
                    "tool_results": merged_tool_results,
                    "summary": synthesis,
                    "mini_outcomes": outcomes,
                },
            },
            "active_agents": [],
        }

    aggregator_node.__name__ = f"{parent_name}_aggregator"
    return aggregator_node


# ── Helpers ────────────────────────────────────────────────────


def _collect_outcomes(state: dict[str, Any], parent_name: str) -> list[dict[str, Any]]:
    """Filter ``mini_agent_outcomes`` by the parent's prefix, preserving id order."""
    raw = state.get("mini_agent_outcomes") or {}
    prefix = f"{parent_name}#"
    matches: list[tuple[str, dict[str, Any]]] = [(key, value) for key, value in raw.items() if key.startswith(prefix)]
    # Sort by mini_id to give the prompt a stable ordering.
    matches.sort(key=lambda kv: kv[1].get("mini_id") or kv[0])
    return [v for _k, v in matches]


def _render_outcome_block(outcomes: list[dict[str, Any]]) -> str:
    """Render the per-outcome bullet list used inside the aggregator prompt."""
    lines: list[str] = []
    for outcome in outcomes:
        status = outcome.get("status", "?")
        description = outcome.get("sub_task_description", "(unknown)")
        if status == "ok":
            body = (outcome.get("summary") or "").strip() or "(no summary)"
        else:
            body = (outcome.get("error") or "").strip() or status
        lines.append(f"  - [{status}] {description}: {body}")
    return "\n".join(lines)


def _merge_successful_tool_results(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``tool_results`` from successful outcomes only.

    Failed/timed-out outcomes contribute to the prompt but never to
    the parent's ``mcp_context`` slot — this prevents partial junk
    from leaking into downstream agents in a sequential pipeline.
    """
    merged: dict[str, Any] = {}
    for outcome in outcomes:
        if outcome.get("status") != "ok":
            continue
        for k, v in (outcome.get("tool_results") or {}).items():
            merged[k] = v
    return merged


def _short_circuit_message(
    *,
    parent_name: str,
    outcomes: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Emit a deterministic error AIMessage with no LLM call."""
    from ..observability import make_event_message

    text = f"Sorry, I couldn't complete this request: {reason}"
    aggregated_event = make_event_message(
        "mini_agent.aggregated",
        {
            "parent": parent_name,
            "outcomes": [{"mini_id": o.get("mini_id", ""), "status": o.get("status", "")} for o in outcomes],
        },
    )
    return {
        "messages": [aggregated_event, AIMessage(content=text, name=parent_name)],
        "mcp_context": {
            parent_name: {
                "tool_results": {},
                "summary": text,
                "mini_outcomes": outcomes,
            },
        },
        "active_agents": [],
    }


def _summarise_failures(outcomes: list[dict[str, Any]]) -> str:
    """Build a one-line cause-of-death summary for the all-failed case."""
    if not outcomes:
        return "no sub-task results were produced"
    buckets: dict[str, list[str]] = {"failed": [], "timeout": []}
    for o in outcomes:
        status = o.get("status", "?")
        if status in buckets:
            buckets[status].append(o.get("sub_task_description") or o.get("mini_id") or "?")
    fragments: list[str] = []
    if buckets["timeout"]:
        fragments.append("timed out: " + ", ".join(buckets["timeout"]))
    if buckets["failed"]:
        fragments.append("failed: " + ", ".join(buckets["failed"]))
    return "; ".join(fragments) or "all sub-tasks failed"


def _fallback_synthesis(outcomes: list[dict[str, Any]]) -> str:
    """When the synthesis LLM call fails, hand the user a deterministic
    summary built from the outcomes we have.  Better than crashing.
    """
    successful = [o for o in outcomes if o.get("status") == "ok"]
    if not successful:
        return f"Sorry, I couldn't complete this request: {_summarise_failures(outcomes)}"
    parts = ["Here is what I found:"]
    for o in successful:
        desc = o.get("sub_task_description", "")
        summary = (o.get("summary") or "").strip()
        if summary:
            parts.append(f"- {desc}: {summary}")
        else:
            parts.append(f"- {desc}")
    return "\n".join(parts)
