"""Mini-agent decomposition step.

The decomposer is a deterministic structured-output LLM call that
decides whether the parent agent's current request can be split into
two-or-more INDEPENDENT sub-tasks suitable for parallel mini-agent
execution.  When ``should_fork=False`` the parent stays on its
single-loop path; when ``True`` the graph fans out to N mini-agents
and an aggregator synthesises their outcomes.

Single-responsibility: this module owns the decomposition logic
ONLY.  The actual mini-agent runtime lives in ``mini_agent_node.py``;
the post-fork synthesis lives in ``mini_agent_aggregator.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config.schema import OrchidAgentConfig
from ..core.mcp import OrchidMCPClient
from ..core.state import OrchidAgentState, OrchidAuthContext

logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_DECOMPOSER_PROMPT",
    "MiniAgentDecomposer",
    "MiniAgentDecomposition",
    "MiniAgentDecompositionError",
    "MiniAgentSubTask",
    "maybe_decompose",
]


# ── Default prompt template (spec §9) ──────────────────────────


DEFAULT_DECOMPOSER_PROMPT = """\
You are the decomposition step for the "{agent_name}" agent.
Agent description: {agent_description}
Agent prompt: {agent_prompt}
Available tools: {tool_inventory}

The user has asked: {user_query}
Conversation history (last {history_max_turns} turns): {history}

Decide whether this request decomposes into INDEPENDENT sub-tasks that
can be processed in parallel by separate mini-agents. A sub-task is
INDEPENDENT iff:
  - it can begin without the result of any other sub-task
  - completing it requires its own multi-step tool-calling loop
    (a single tool call should usually NOT be a mini-agent — that
     belongs in parallel_tools)

If the request is already a single coherent task, return should_fork=false.
Otherwise emit 2..{max_count} sub-tasks. For each sub-task:
  - id: "mini_0", "mini_1", ...
  - description: short user-facing label (≤ 80 chars)
  - instruction: focused system-prompt suffix appended to the agent's prompt
  - allowed_tools: minimal subset of available tools needed for this sub-task
  - rationale: one sentence explaining independence
"""


# ── Models ─────────────────────────────────────────────────────


class MiniAgentSubTask(BaseModel):
    """One independent sub-task emitted by the decomposer.

    The fields mirror the prompt template in :data:`DEFAULT_DECOMPOSER_PROMPT`
    one-to-one.  ``allowed_tools`` is enforced post-decomposition by
    :class:`MiniAgentDecomposer` according to the parent agent's
    ``mini_agent.tool_allowlist_mode`` setting.
    """

    id: str = Field(description='Stable identifier, e.g. "mini_0".')
    description: str = Field(description="Short user-facing label (≤ 80 chars).")
    instruction: str = Field(
        description="Focused system-prompt suffix appended to the agent's prompt.",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Minimal subset of tools the sub-task may call.",
    )
    rationale: str = Field(
        description="One sentence explaining why this sub-task is independent.",
    )

    model_config = ConfigDict(extra="forbid")


class MiniAgentDecomposition(BaseModel):
    """Top-level decomposer result — fork? and (if so) sub-tasks."""

    should_fork: bool = Field(
        description="True iff the request decomposes into independent sub-tasks.",
    )
    sub_tasks: list[MiniAgentSubTask] = Field(
        default_factory=list,
        description="0 if should_fork=false; otherwise 2..max_count entries.",
    )
    reasoning: str = Field(
        default="",
        description="Debug-only explanation surfaced in the trace.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_subtask_count(self) -> MiniAgentDecomposition:
        """Force the should_fork flag and sub_tasks list to agree.

        - ``should_fork=False`` → ``sub_tasks`` MUST be empty.
        - ``should_fork=True`` → at least 2 sub-tasks (forking 1 is
          pointless).  Upper bound is enforced by
          :meth:`MiniAgentDecomposer.decompose` against the parent
          agent's per-agent ``max_count`` configuration.
        """
        if not self.should_fork:
            if self.sub_tasks:
                raise ValueError(
                    "MiniAgentDecomposition: should_fork=False but sub_tasks is non-empty",
                )
            return self
        if len(self.sub_tasks) < 2:
            raise ValueError(
                "MiniAgentDecomposition: should_fork=True requires at least 2 sub-tasks",
            )
        seen_ids = set()
        for st in self.sub_tasks:
            if st.id in seen_ids:
                raise ValueError(
                    f"MiniAgentDecomposition: duplicate sub_task id '{st.id}'",
                )
            seen_ids.add(st.id)
        return self


class MiniAgentDecompositionError(Exception):
    """Raised when the decomposer's output violates the parent's allowlist
    rules.  The parent ``GenericAgent`` catches this and short-circuits
    to an "all-failed" path (no minis dispatched, error AIMessage).
    """


# ── The decomposer itself ──────────────────────────────────────


class MiniAgentDecomposer:
    """Run the decomposer LLM call against an arbitrary chat model.

    Pure: no I/O outside of a single ``ainvoke`` per call.  The chat
    model is injected by the parent (built from the agent's
    ``mini_agent.decomposer_model`` or the agent's own ``llm.model``
    via ``llm_factory.build_chat_model``).

    Owns:

    1. Prompt rendering from the configurable template.
    2. Structured-output binding via ``with_structured_output``.
    3. Post-validation of the decomposer's ``allowed_tools`` against
       the parent's full tool inventory per the
       ``tool_allowlist_mode`` setting.
    """

    def __init__(self, *, agent_config: OrchidAgentConfig, chat_model: Any) -> None:
        self._config = agent_config
        self._chat_model = chat_model

    async def decompose(
        self,
        *,
        user_query: str,
        conversation_history: list[dict[str, str]] | None,
        tool_inventory: list[str],
        history_max_turns: int = 20,
    ) -> MiniAgentDecomposition:
        """Run the decomposer LLM call and return a validated decomposition.

        Parameters
        ----------
        user_query
            The current user request (post-reformulation).
        conversation_history
            Optional list of ``{"role": ..., "content": ...}`` dicts —
            included verbatim in the prompt so the decomposer sees
            the same multi-turn context the agent does.  ``None`` /
            empty → "(no history)".
        tool_inventory
            Sorted list of every tool name the parent agent can call.
            The decomposer's ``allowed_tools`` is validated against
            this list per ``tool_allowlist_mode``.
        history_max_turns
            Hint for the prompt — mirrors the supervisor's
            ``OrchidSupervisorConfig.history_max_turns``.  Has no
            effect on actual truncation (the caller already truncated).

        Raises
        ------
        MiniAgentDecompositionError
            When the decomposer produces an ``allowed_tools`` list
            that violates the parent's ``tool_allowlist_mode``
            (``strict`` mode only).
        """
        prompt_template = self._config.mini_agent.decomposer_prompt or DEFAULT_DECOMPOSER_PROMPT
        rendered = prompt_template.format(
            agent_name=self._config.name,
            agent_description=self._config.description,
            agent_prompt=self._config.prompt,
            tool_inventory=", ".join(tool_inventory) if tool_inventory else "(none)",
            user_query=user_query,
            history=_render_history(conversation_history),
            history_max_turns=history_max_turns,
            max_count=self._config.mini_agent.max_count,
        )

        # ``with_structured_output`` returns an LLM that yields a
        # validated ``MiniAgentDecomposition`` instance directly — no
        # JSON parsing required at the call site.  Same pattern as
        # the supervisor's ``OrchidRoutingDecision``.
        structured_llm = self._chat_model.with_structured_output(MiniAgentDecomposition)
        decomposition: MiniAgentDecomposition = await structured_llm.ainvoke(
            [{"role": "system", "content": rendered}],
        )

        # Cap sub-task count to the per-agent ``max_count`` ceiling.
        # The Pydantic validator already rejected the "exactly 1"
        # degenerate case; this enforces the upper bound which the
        # model itself cannot know without hard-coding.
        cap = self._config.mini_agent.max_count
        if len(decomposition.sub_tasks) > cap:
            raise MiniAgentDecompositionError(
                f"decomposer returned {len(decomposition.sub_tasks)} sub-tasks but "
                f"max_count is {cap} for agent '{self._config.name}'",
            )

        if decomposition.should_fork:
            self._enforce_tool_allowlist(decomposition, tool_inventory)

        return decomposition

    def _enforce_tool_allowlist(
        self,
        decomposition: MiniAgentDecomposition,
        tool_inventory: list[str],
    ) -> None:
        """Validate every sub-task's ``allowed_tools`` per the parent's mode.

        - ``strict`` (default): each tool name must exist in the
          parent's full inventory; otherwise raise.
        - ``parent_full``: ignore ``allowed_tools`` entirely (the
          mini node will substitute the full inventory).
        - ``inferred``: empty ``allowed_tools`` falls back to the
          full inventory with a warning; non-empty behaves like
          strict.
        """
        mode = self._config.mini_agent.tool_allowlist_mode
        inventory = set(tool_inventory)
        if mode == "parent_full":
            return

        for sub_task in decomposition.sub_tasks:
            if not sub_task.allowed_tools:
                if mode == "inferred":
                    logger.warning(
                        "[%s] mini-agent '%s' has empty allowed_tools — "
                        "falling back to full parent inventory (inferred mode)",
                        self._config.name,
                        sub_task.id,
                    )
                    continue
                # strict mode forbids empty too — be explicit.
                raise MiniAgentDecompositionError(
                    f"sub_task '{sub_task.id}' has empty allowed_tools but "
                    f"tool_allowlist_mode='strict' (agent '{self._config.name}')",
                )

            unknown = [t for t in sub_task.allowed_tools if t not in inventory]
            if unknown:
                raise MiniAgentDecompositionError(
                    f"sub_task '{sub_task.id}' references unknown tools {unknown!r} "
                    f"not in parent '{self._config.name}' inventory",
                )

    def resolve_tool_subset(
        self,
        *,
        sub_task: MiniAgentSubTask,
        tool_inventory: list[str],
    ) -> list[str]:
        """Resolve the effective tool subset for a single mini.

        Mirrors the precedence used by :meth:`_enforce_tool_allowlist`
        but returns the concrete name list the mini node should pass
        to ``AgenticLoop(tool_subset=...)``.
        """
        mode = self._config.mini_agent.tool_allowlist_mode
        if mode == "parent_full":
            return list(tool_inventory)
        if mode == "inferred" and not sub_task.allowed_tools:
            return list(tool_inventory)
        return list(sub_task.allowed_tools)


# ── Graph-level decomposer hook ───────────────────────────────


async def maybe_decompose(
    *,
    agent_config: OrchidAgentConfig,
    chat_model: Any,
    mcp_clients: list[OrchidMCPClient],
    auth: OrchidAuthContext,
    state: OrchidAgentState,
) -> dict[str, Any] | None:
    """Run the decomposer for ``agent_config`` if its mini-agent block is enabled.

    Returns one of:

    - ``None`` — ``mini_agent.enabled`` is False, the decomposer
      decided not to fork, the agent has no chat model, or the LLM
      call failed in a recoverable way.  The graph wrapper then
      proceeds to call ``agent.run(state)`` as normal.
    - State update with ``mini_agent_decisions[parent_name]`` and a
      ``mini_agent.decomposed`` event ``SystemMessage`` — the
      decomposer chose to fork; the graph's conditional edge fans
      out into mini-agents.
    - State update with an error ``AIMessage`` named after the
      parent — the decomposer's output violated the allowlist
      (``MiniAgentDecompositionError``).  No minis dispatch.

    SOLID note: this function is the SINGLE entry point for invoking
    the decomposer from a parent_agent_node, regardless of the
    parent agent's concrete class.  ``GenericAgent`` and custom
    subclasses both go through it via the graph's wrapper closure
    — neither needs to inline the decomposer logic into its
    ``run()`` method.
    """
    # Local import to avoid the ``OrchidAgent.run()`` ↔ event-helpers
    # chain at import time (events only exist for the streaming surface).
    from ..observability import make_event_message

    if not agent_config.mini_agent.enabled:
        return None
    if chat_model is None:
        return None

    # Build the decomposer's chat model: parent's by default,
    # overridden when ``mini_agent.decomposer_model`` differs from
    # the parent's effective ``llm.model``.
    parent_model = (agent_config.llm.model if agent_config.llm else None) or ""
    decomposer_model = agent_config.mini_agent.decomposer_model
    if decomposer_model and decomposer_model != parent_model:
        from ..llm_factory import build_chat_model

        decomposer_chat = build_chat_model(decomposer_model, temperature=0.0)
    else:
        decomposer_chat = chat_model

    # Render the parent's full tool inventory once — names + a
    # compact description list for the prompt and a ground-truth set
    # for allowlist enforcement.
    try:
        inventory = await _render_tool_inventory(
            agent_config=agent_config,
            mcp_clients=mcp_clients,
            auth=auth,
        )
    except Exception as exc:
        logger.warning(
            "[%s] decomposer: tool inventory rendering failed (%s) — running without tools list",
            agent_config.name,
            exc,
        )
        inventory = []

    # Local import — the agent module imports this one, so go the
    # other way through ``OrchidAgent`` directly to avoid any cycle.
    from ..core.agent import OrchidAgent

    history = OrchidAgent.extract_conversation_history(state)
    user_query = OrchidAgent.extract_user_query(state)

    decomposer = MiniAgentDecomposer(agent_config=agent_config, chat_model=decomposer_chat)
    try:
        decomposition = await decomposer.decompose(
            user_query=user_query,
            conversation_history=history,
            tool_inventory=inventory,
        )
    except MiniAgentDecompositionError as exc:
        logger.warning(
            "[%s] decomposer rejected: %s — short-circuiting with error AIMessage",
            agent_config.name,
            exc,
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"[{agent_config.name.title()} Agent] I couldn't break this request "
                        f"into independent sub-tasks: {exc}"
                    ),
                    name=agent_config.name,
                ),
            ],
            "mcp_context": {agent_config.name: {"summary": str(exc)}},
        }
    except Exception as exc:
        logger.warning(
            "[%s] decomposer LLM call failed (%s) — falling back to normal flow",
            agent_config.name,
            exc,
        )
        return None

    if not decomposition.should_fork:
        logger.info(
            "[%s] decomposer: should_fork=False (%s)",
            agent_config.name,
            (decomposition.reasoning or "")[:120] or "(no reason given)",
        )
        return None

    # Resolve per-sub-task tool subsets so the fork router can stamp
    # them on each ``Send`` payload.
    sub_task_payloads: list[dict[str, Any]] = []
    for sub_task in decomposition.sub_tasks:
        tool_subset = decomposer.resolve_tool_subset(
            sub_task=sub_task,
            tool_inventory=inventory,
        )
        payload = sub_task.model_dump()
        payload["resolved_tool_subset"] = tool_subset
        sub_task_payloads.append(payload)

    decision_dump = decomposition.model_dump()
    decision_dump["sub_tasks"] = sub_task_payloads

    logger.info(
        "[%s] decomposer: should_fork=True (%d sub-tasks)",
        agent_config.name,
        len(decomposition.sub_tasks),
    )

    # Emit the ``mini_agent.decomposed`` event so the streaming
    # router can surface decomposition activity to the client.
    event = make_event_message(
        "mini_agent.decomposed",
        {
            "parent": agent_config.name,
            "count": len(decomposition.sub_tasks),
            "sub_tasks": [{"id": s.id, "description": s.description} for s in decomposition.sub_tasks],
        },
    )

    return {
        "mini_agent_decisions": {agent_config.name: decision_dump},
        "messages": [event],
    }


async def _render_tool_inventory(
    *,
    agent_config: OrchidAgentConfig,
    mcp_clients: list[OrchidMCPClient],
    auth: OrchidAuthContext,
) -> list[str]:
    """Compute the parent's full tool-name inventory (built-ins + MCP)."""
    from ..config.tool_registry import get_tool

    names: list[str] = []
    seen: set[str] = set()

    # Built-in tools that survive registry lookup.
    for tool_name in agent_config.tools:
        try:
            get_tool(tool_name)
        except KeyError:
            continue
        if tool_name not in seen:
            names.append(tool_name)
            seen.add(tool_name)

    # MCP tools — declared and discovered.  Cache hits are fast; this
    # also runs again inside the parent's normal ``run()`` flow so
    # the caches remain warm either way.
    if agent_config.mcp_servers and mcp_clients:
        from .mcp_dispatcher import MCPDispatcher

        dispatcher = MCPDispatcher(mcp_clients=mcp_clients, server_configs=agent_config.mcp_servers)
        caps = await dispatcher.render_capabilities(auth, agent_name=agent_config.name)
        for raw in caps.raw_tools:
            tool_name = raw.get("name")
            if tool_name and tool_name not in seen:
                names.append(tool_name)
                seen.add(tool_name)

    return names


# ── Helpers ────────────────────────────────────────────────────


def _render_history(history: list[dict[str, str]] | None) -> str:
    """Render conversation history into a compact, prompt-friendly form."""
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if not content:
            continue
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"
