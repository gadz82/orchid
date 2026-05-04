"""Per-agent mini-agent (self-clone) configuration — opt-in.

When ``OrchidMiniAgentConfig.enabled`` is ``True`` on an agent the
graph builder synthesises three nodes for that agent:

- ``{name}_agent`` — the parent (``GenericAgent``) runs the
  decomposer LLM call before its existing 6-step flow.  When the
  decomposer returns ``should_fork=True`` the parent emits a state
  update WITHOUT an ``AIMessage`` so the conditional edge can fan
  out into mini-agents instead of returning to the supervisor.
- ``{name}_mini`` — a single node invoked once per sub-task via
  ``Send``; runs an ``AgenticLoop`` against the parent's MCP clients
  with a curated tool subset and a focused system prompt.
- ``{name}_aggregator`` — synthesises the per-mini outcomes back
  into one ``AIMessage`` and writes ``mcp_context[parent_name]``.

Agents that don't opt in keep their current single-node shape; the
builder must not synthesise the extra nodes for them (zero overhead).

See ``.knowledge/mini-agents-implementation-spec.md`` §5–§12 for the
full contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrchidMiniAgentConfig(BaseModel):
    """Per-agent mini-agent (self-clone) configuration.  Opt-in."""

    # Master switch — when False (default) the agent runs its current
    # single-node flow and the builder synthesises no extra nodes.
    enabled: bool = False

    # Per-agent ceiling.  The decomposer's ``MiniAgentDecomposition``
    # validator caps ``len(sub_tasks) <= max_count``.  Lower bound 2
    # because forking a single sub-task is pointless; upper bound 8
    # to keep the LangGraph fan-out bounded by configuration alone.
    max_count: int = Field(default=3, ge=2, le=8)

    # Optional override for the decomposer's chat model.  Falls back
    # to the parent agent's ``llm.model`` when ``None``.
    decomposer_model: str | None = None

    # Hard timeout per mini, applied via ``asyncio.wait_for`` around
    # the inner agentic loop.  Bounded between 5 s and 600 s.
    timeout_seconds: int = Field(default=60, ge=5, le=600)

    # Tool exposure mode (spec §9).  ``strict`` (default) requires
    # every name in the decomposer's ``allowed_tools`` to exist in
    # the parent's full inventory; ``parent_full`` ignores
    # ``allowed_tools`` entirely (debug / escape hatch);
    # ``inferred`` is ``strict`` except an empty ``allowed_tools``
    # falls back to the full parent inventory with a warning.
    tool_allowlist_mode: Literal["strict", "parent_full", "inferred"] = "strict"

    # When True, mini-agent inner ``on_chat_model_stream`` events
    # propagate to the streaming router with a per-mini prefix.
    # Default False — internal token streams are suppressed and
    # only the four ``mini_agent.*`` lifecycle events surface.
    # Wired in PR-3; declared here so the YAML schema is stable.
    stream_inner_tokens: bool = False

    # Optional prompt overrides — when ``None`` the decomposer and
    # aggregator use the default templates baked into the relevant
    # modules (see ``mini_agent_decomposer.DEFAULT_DECOMPOSER_PROMPT``
    # and ``mini_agent_aggregator.DEFAULT_AGGREGATOR_PROMPT``).
    decomposer_prompt: str | None = None
    aggregator_prompt: str | None = None

    #: Optional override for the per-mini system prompt assembled by
    #: :func:`orchid_ai.agents.mini_agent_node._build_mini_system_prompt`.
    #: Resolved with :py:meth:`str.format` against the placeholders
    #: ``{parent_prompt}``, ``{instruction}``, ``{tool_list}`` (a
    #: newline-joined ``- name: description`` bullet list).  ``None``
    #: uses the built-in template.  See the
    #: ``examples/prompt-customization`` example for a worked usage.
    system_prompt_template: str | None = None

    # Reject unknown YAML keys so typos surface immediately.
    model_config = ConfigDict(extra="forbid")
