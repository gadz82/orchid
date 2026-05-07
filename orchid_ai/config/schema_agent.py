"""Agent + root configuration models, plus the defaults-merging pass."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .mcp_gateway import OrchidMCPGatewayConfig
from .schema_events import OrchidEventsConfig
from .schema_guardrails import OrchidGuardrailsConfig
from .schema_llm import OrchidLLMConfig
from .schema_mcp import OrchidMCPServerConfig, OrchidToolConfig
from .schema_mini_agent import OrchidMiniAgentConfig
from .schema_prompts import OrchidAgentPromptConfig
from .schema_rag import OrchidRAGConfig, OrchidRAGDefaultsConfig
from .schema_skills import (
    OrchidAgentSkillConfig,
    OrchidBuiltinToolConfig,
    OrchidOrchestratorSkillConfig,
)
from .schema_supervisor import ExecutionHints, OrchidSupervisorConfig


class OrchidAgentConfig(BaseModel):
    """
    Configuration for a single agent.

    When ``class_path`` is ``None``, the ``GenericAgent`` is used.
    When set, it must be either:
      - A dotted Python import path (e.g. ``myapp.agents.support.SupportAgent``)
      - A short name registered in the agent class registry
    """

    # Set programmatically by the loader (= the YAML dict key)
    name: str = ""

    description: str
    prompt: str

    # ``class`` in YAML → ``class_path`` in Python (reserved word)
    class_path: str | None = Field(default=None, alias="class")

    rag: OrchidRAGConfig = Field(default_factory=OrchidRAGConfig)
    mcp_servers: list[OrchidMCPServerConfig] = Field(default_factory=list)
    llm: OrchidLLMConfig | None = None
    execution_hints: ExecutionHints = Field(default_factory=ExecutionHints)

    # Built-in tools available to this agent
    tools: list[str] = Field(default_factory=list)

    # Agent-level skills — multi-step workflows within this agent
    skills: dict[str, OrchidAgentSkillConfig] = Field(default_factory=dict)

    # Per-agent guardrails
    guardrails: OrchidGuardrailsConfig = Field(default_factory=OrchidGuardrailsConfig)

    # Recursive nesting — sub-agents under this agent
    children: dict[str, OrchidAgentConfig] | None = None

    # Computed at validation — tool names whose results are injected to RAG
    injectable_tools: set[str] = Field(default_factory=set, exclude=True)

    # Computed at validation — tool name → effective TTL (seconds) for RAG cache
    # Only includes tools with inject_to_rag=True AND effective TTL > 0
    injectable_tool_ttls: dict[str, int] = Field(default_factory=dict, exclude=True)

    # Computed at validation — tool names that require human approval (HITL)
    approval_tools: set[str] = Field(default_factory=set, exclude=True)

    # Computed at validation — built-in tool names whose
    # ``OrchidBuiltinToolConfig.parallel_safe`` is explicitly ``True``.
    # MCP tools resolve their per-tool override at runtime against the
    # nested ``mcp_servers[i].tools[j].parallel_safe`` field plus the
    # MCP ``readOnlyHint`` annotation, so they don't need a precomputed
    # set; built-ins have no annotation fallback so we precompute here
    # (built-in tool configs live on ``OrchidAgentsConfig.tools`` which
    # the running agent cannot reach directly).
    parallel_safe_builtin_tools: set[str] = Field(default_factory=set, exclude=True)

    # Computed at validation — resolved ``OrchidBuiltinToolConfig`` per
    # tool name referenced by this agent.  Cached here so
    # :meth:`effective_rag` can look up per-tool RAG overrides at
    # runtime without needing the global ``OrchidAgentsConfig.tools``
    # dict (the agent only knows tool names, not the underlying
    # config objects).
    builtin_tool_configs: dict[str, OrchidBuiltinToolConfig] = Field(default_factory=dict, exclude=True)

    # Opt-in parallel tool-call dispatch within a single
    # agentic round.  When ``True``, the agentic loop partitions the
    # LLM's tool_calls into a parallel batch (gathered via
    # ``asyncio.gather``) and a sequential tail.  Per-tool safety is
    # resolved from ``OrchidToolConfig.parallel_safe`` / the MCP
    # ``readOnlyHint`` annotation / ``OrchidBuiltinToolConfig.parallel_safe``
    # — see ``GenericAgent._resolve_parallel_safety`` for the precedence
    # rules.  Defaults to ``False`` to preserve today's serial behaviour.
    parallel_tools: bool = False

    # Opt-in mini-agent (self-clone) configuration.  When
    # ``mini_agent.enabled=True``, the graph builder synthesises
    # ``{name}_mini`` and ``{name}_aggregator`` nodes alongside the
    # normal ``{name}_agent`` parent node.  Defaults to disabled —
    # zero overhead for agents that don't opt in.  Nesting is
    # forbidden: child agents cannot enable mini-agents (no recursion
    # within a single supervisor turn).  See ``schema_mini_agent.py``
    # for field semantics.
    mini_agent: OrchidMiniAgentConfig = Field(default_factory=OrchidMiniAgentConfig)

    # Customisable templates for the agentic-loop system prompt — the
    # six section headers + per-resource bodies + truncation knobs.
    # See :class:`OrchidAgentPromptConfig` for the full field list and
    # placeholder contracts.
    prompt_sections: OrchidAgentPromptConfig = Field(default_factory=OrchidAgentPromptConfig)

    model_config = {"populate_by_name": True}

    def effective_rag(self, tool_name: str) -> OrchidRAGConfig:
        """Return the RAG config that should govern ``tool_name``.

        Looks up the tool first in this agent's MCP server tools, then
        in the cached built-in tool configs.  When the tool sets a
        ``rag:`` block, the agent's full RAG config is used as the
        base and the tool's *explicitly set* fields overlay onto it
        via a deep merge — so a tool can override just one nested
        knob (``retrieval.strategy``, ``ingestion.chunk_size``, …)
        without restating every other field.  When no override is
        set, returns ``self.rag`` unchanged.
        """
        tool_rag: OrchidRAGConfig | None = None

        for server in self.mcp_servers:
            for tool in server.tools:
                if isinstance(tool, OrchidToolConfig) and tool.name == tool_name and tool.rag is not None:
                    tool_rag = tool.rag
                    break
            if tool_rag is not None:
                break

        if tool_rag is None:
            builtin_cfg = self.builtin_tool_configs.get(tool_name)
            if builtin_cfg is not None and builtin_cfg.rag is not None:
                tool_rag = builtin_cfg.rag

        if tool_rag is None:
            return self.rag

        base = self.rag.model_dump()
        overlay = tool_rag.model_dump(exclude_unset=True)
        merged = _deep_merge(base, overlay)
        return OrchidRAGConfig.model_validate(merged)


class OrchidDefaultsConfig(BaseModel):
    """Top-level defaults inherited by every agent.

    ``cache_enabled`` activates a global in-memory LLM response cache
    via LangChain's ``set_llm_cache(InMemoryCache())``.  Identical
    prompts return cached results, reducing latency and cost.  Cache
    lives for the process lifetime (reset on restart).

    Example YAML::

        defaults:
          cache_enabled: true
          llm:
            model: gemini/gemini-2.5-flash
    """

    llm: OrchidLLMConfig = Field(default_factory=OrchidLLMConfig)
    rag: OrchidRAGDefaultsConfig = Field(default_factory=OrchidRAGDefaultsConfig)
    cache_enabled: bool = False


class OrchidAgentsConfig(BaseModel):
    """
    Root configuration loaded from agents.yaml.

    After validation, each ``OrchidAgentConfig`` has its defaults merged in
    and its ``name`` set from the YAML dict key.
    """

    version: str = "1"
    defaults: OrchidDefaultsConfig = Field(default_factory=OrchidDefaultsConfig)

    # Global built-in tool declarations
    tools: dict[str, OrchidBuiltinToolConfig] = Field(default_factory=dict)

    # Orchestrator-level skills — cross-agent workflows
    skills: dict[str, OrchidOrchestratorSkillConfig] = Field(default_factory=dict)

    # Supervisor configuration — prompt customization
    supervisor: OrchidSupervisorConfig = Field(default_factory=OrchidSupervisorConfig)

    # Global guardrails — applied to all requests
    guardrails: OrchidGuardrailsConfig = Field(default_factory=OrchidGuardrailsConfig)

    # MCP gateway exposure — tool title/description overrides + MCP Prompts.
    # Consumed by any MCP-facing gateway (e.g. ``orchid-mcp``) to customise
    # how Orchid is presented to the host LLM.  Purely declarative — the
    # framework library does not render or validate tool/prompt semantics,
    # only the shape of the data.
    mcp_gateway: OrchidMCPGatewayConfig = Field(default_factory=OrchidMCPGatewayConfig)

    agents: dict[str, OrchidAgentConfig] = Field(default_factory=dict)

    # Pollen + Bloom — event-driven activation layer.  ``None`` (the
    # default) keeps the framework's existing zero-overhead behaviour:
    # no producers, no processors, no events tables touched.  An
    # explicit ``events:`` block opts in.
    events: OrchidEventsConfig | None = None

    @model_validator(mode="after")
    def _apply_defaults_and_names(self) -> OrchidAgentsConfig:
        """Merge defaults into each agent and set names recursively."""
        for agent_name, agent in self.agents.items():
            _apply_defaults(agent, agent_name, self.defaults, self.tools)
        return self


def _apply_defaults(
    agent: OrchidAgentConfig,
    name: str,
    defaults: OrchidDefaultsConfig,
    global_tools: dict[str, OrchidBuiltinToolConfig] | None = None,
) -> None:
    """Recursively apply default values and set agent names."""
    # Set name from dict key
    agent.name = name

    # Merge LLM defaults
    if agent.llm is None:
        agent.llm = defaults.llm.model_copy()

    # Merge RAG defaults (only if not explicitly set to non-default)
    if agent.rag.k == 5 and defaults.rag.k != 5:
        agent.rag.k = defaults.rag.k
    if agent.rag.enabled and not defaults.rag.enabled:
        agent.rag.enabled = defaults.rag.enabled
    if agent.rag.rag_ttl == 0 and defaults.rag.rag_ttl != 0:
        agent.rag.rag_ttl = defaults.rag.rag_ttl
    if agent.rag.max_context_chars is None:
        agent.rag.max_context_chars = defaults.rag.max_context_chars

    # Merge retrieval block — strategy + transformers inherit
    # independently so an agent can override one without losing the
    # other.  ``strategy`` falls back to ``"simple"`` when neither
    # agent nor defaults set it; ``query_transformers`` falls back to
    # ``[]`` (no transformers) on the same path.
    if agent.rag.retrieval.strategy is None:
        agent.rag.retrieval.strategy = defaults.rag.retrieval.strategy or "simple"
    if agent.rag.retrieval.query_transformers is None:
        agent.rag.retrieval.query_transformers = list(defaults.rag.retrieval.query_transformers or [])
    if not agent.rag.retrieval.metadata_filters and defaults.rag.retrieval.metadata_filters:
        agent.rag.retrieval.metadata_filters = dict(defaults.rag.retrieval.metadata_filters)

    # Merge transformer prompt overrides — each scalar field inherits
    # independently when the agent leaves it ``None``, so an agent can
    # override one transformer's prompt without losing the others.
    _merge_transformer_prompts(agent.rag.retrieval, defaults.rag.retrieval)

    # Merge ingestion block — ``strategy`` falls back to ``"recursive"``
    # when neither side sets it; chunk knobs inherit independently.
    if agent.rag.ingestion.strategy is None:
        agent.rag.ingestion.strategy = defaults.rag.ingestion.strategy or "recursive"
    if agent.rag.ingestion.chunk_size == 1000 and defaults.rag.ingestion.chunk_size != 1000:
        agent.rag.ingestion.chunk_size = defaults.rag.ingestion.chunk_size
    if agent.rag.ingestion.chunk_overlap == 200 and defaults.rag.ingestion.chunk_overlap != 200:
        agent.rag.ingestion.chunk_overlap = defaults.rag.ingestion.chunk_overlap
    if agent.rag.ingestion.parent_chunk_size == 0 and defaults.rag.ingestion.parent_chunk_size != 0:
        agent.rag.ingestion.parent_chunk_size = defaults.rag.ingestion.parent_chunk_size
    if agent.rag.ingestion.parent_chunk_overlap == 200 and defaults.rag.ingestion.parent_chunk_overlap != 200:
        agent.rag.ingestion.parent_chunk_overlap = defaults.rag.ingestion.parent_chunk_overlap
    if not agent.rag.ingestion.post_processors and defaults.rag.ingestion.post_processors:
        agent.rag.ingestion.post_processors = list(defaults.rag.ingestion.post_processors)

    # Collect injectable MCP tool names + TTLs
    agent_ttl = agent.rag.rag_ttl
    for server in agent.mcp_servers:
        for tool in server.tools:
            if isinstance(tool, OrchidToolConfig) and tool.inject_to_rag:
                agent.injectable_tools.add(tool.name)
                effective_ttl = tool.rag_ttl if tool.rag_ttl is not None else agent_ttl
                if effective_ttl > 0:
                    agent.injectable_tool_ttls[tool.name] = effective_ttl

    # Collect injectable built-in tool names + TTLs
    if global_tools:
        for tool_name in agent.tools:
            tool_cfg = global_tools.get(tool_name)
            if tool_cfg and tool_cfg.inject_to_rag:
                key = f"builtin_{tool_name}"
                agent.injectable_tools.add(key)
                effective_ttl = tool_cfg.rag_ttl if tool_cfg.rag_ttl is not None else agent_ttl
                if effective_ttl > 0:
                    agent.injectable_tool_ttls[key] = effective_ttl

    # Collect tools requiring human approval (HITL)
    for server in agent.mcp_servers:
        for tool in server.tools:
            if isinstance(tool, OrchidToolConfig) and tool.requires_approval:
                agent.approval_tools.add(tool.name)
    if global_tools:
        for tool_name in agent.tools:
            tool_cfg = global_tools.get(tool_name)
            if tool_cfg and tool_cfg.requires_approval:
                agent.approval_tools.add(tool_name)

    # Collect built-in tools whose ``parallel_safe`` is explicitly
    # ``True`` — used by the agentic loop's parallel-dispatch path
    # when the agent has ``parallel_tools: true``.
    if global_tools:
        for tool_name in agent.tools:
            tool_cfg = global_tools.get(tool_name)
            if tool_cfg and tool_cfg.parallel_safe is True:
                agent.parallel_safe_builtin_tools.add(tool_name)

    # Cache resolved built-in tool configs so
    # ``OrchidAgentConfig.effective_rag(tool_name)`` can look up
    # per-tool RAG overrides at runtime — the agent only knows tool
    # names, not the underlying ``OrchidBuiltinToolConfig`` objects on
    # ``OrchidAgentsConfig.tools``.
    if global_tools:
        for tool_name in agent.tools:
            tool_cfg = global_tools.get(tool_name)
            if tool_cfg is not None:
                agent.builtin_tool_configs[tool_name] = tool_cfg

    # Recurse into children — but reject any child that opts into
    # mini-agents.  Nesting is forbidden by spec §2 to keep the
    # graph topology bounded.  ``mini_agent.enabled`` may only be
    # set on top-level agents, never on children of another agent.
    if agent.children:
        for child_name, child in agent.children.items():
            if child.mini_agent.enabled:
                raise ValueError(
                    f"agent '{name}.{child_name}' has mini_agent.enabled=true — "
                    f"mini-agents may only be enabled on top-level agents (no nesting)."
                )
            _apply_defaults(child, child_name, defaults, global_tools)


def _merge_transformer_prompts(agent_retrieval: object, defaults_retrieval: object) -> None:
    """Inherit unset transformer-prompt overrides from the defaults block.

    Each scalar field is treated independently — leaving any field
    ``None`` on the agent inherits whatever the defaults block sets
    (which is also ``None`` in the typical case, so the transformer
    falls back to its module-level default).  Splitting this out keeps
    :func:`_apply_defaults` readable.
    """
    agent_prompts = agent_retrieval.transformer_prompts  # type: ignore[attr-defined]
    default_prompts = defaults_retrieval.transformer_prompts  # type: ignore[attr-defined]

    if agent_prompts.multi_query is None and default_prompts.multi_query is not None:
        agent_prompts.multi_query = default_prompts.multi_query
    if agent_prompts.decompose is None and default_prompts.decompose is not None:
        agent_prompts.decompose = default_prompts.decompose
    if agent_prompts.reformulate is None and default_prompts.reformulate is not None:
        agent_prompts.reformulate = default_prompts.reformulate
    if agent_prompts.hyde.single is None and default_prompts.hyde.single is not None:
        agent_prompts.hyde.single = default_prompts.hyde.single
    if agent_prompts.hyde.multi is None and default_prompts.hyde.multi is not None:
        agent_prompts.hyde.multi = default_prompts.hyde.multi


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay ``overlay`` onto ``base``, preserving nested keys.

    Used by :meth:`OrchidAgentConfig.effective_rag` to merge a tool's
    explicitly-set ``rag`` fields onto the agent's full RAG dump
    without losing untouched nested values (e.g. an
    ``ingestion: {chunk_size: 500}`` overlay must keep the agent's
    ``ingestion.strategy`` intact).
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
