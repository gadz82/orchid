"""Agent + root configuration models, plus the defaults-merging pass."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .mcp_gateway import OrchidMCPGatewayConfig
from .schema_guardrails import OrchidGuardrailsConfig
from .schema_llm import OrchidLLMConfig
from .schema_mcp import OrchidMCPServerConfig, OrchidToolConfig
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

    # Built-in tools available to this agent (ADR-017)
    tools: list[str] = Field(default_factory=list)

    # Agent-level skills — multi-step workflows within this agent (ADR-017)
    skills: dict[str, OrchidAgentSkillConfig] = Field(default_factory=dict)

    # Per-agent guardrails (ADR-018)
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

    # Phase A — opt-in parallel tool-call dispatch within a single
    # agentic round.  When ``True``, the agentic loop partitions the
    # LLM's tool_calls into a parallel batch (gathered via
    # ``asyncio.gather``) and a sequential tail.  Per-tool safety is
    # resolved from ``OrchidToolConfig.parallel_safe`` / the MCP
    # ``readOnlyHint`` annotation / ``OrchidBuiltinToolConfig.parallel_safe``
    # — see ``GenericAgent._resolve_parallel_safety`` for the precedence
    # rules.  Defaults to ``False`` to preserve today's serial behaviour.
    parallel_tools: bool = False

    model_config = {"populate_by_name": True}


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

    # Global built-in tool declarations (ADR-017)
    tools: dict[str, OrchidBuiltinToolConfig] = Field(default_factory=dict)

    # Orchestrator-level skills — cross-agent workflows (ADR-017)
    skills: dict[str, OrchidOrchestratorSkillConfig] = Field(default_factory=dict)

    # Supervisor configuration — prompt customization
    supervisor: OrchidSupervisorConfig = Field(default_factory=OrchidSupervisorConfig)

    # Global guardrails — applied to all requests (ADR-018)
    guardrails: OrchidGuardrailsConfig = Field(default_factory=OrchidGuardrailsConfig)

    # MCP gateway exposure — tool title/description overrides + MCP Prompts.
    # Consumed by any MCP-facing gateway (e.g. ``orchid-mcp``) to customise
    # how Orchid is presented to the host LLM.  Purely declarative — the
    # framework library does not render or validate tool/prompt semantics,
    # only the shape of the data.
    mcp_gateway: OrchidMCPGatewayConfig = Field(default_factory=OrchidMCPGatewayConfig)

    agents: dict[str, OrchidAgentConfig] = Field(default_factory=dict)

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
    if not defaults.rag.reformulate_queries:
        agent.rag.reformulate_queries = False
    if agent.rag.retriever_type is None:
        agent.rag.retriever_type = defaults.rag.retriever_type
    if agent.rag.max_context_chars is None:
        agent.rag.max_context_chars = defaults.rag.max_context_chars

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

    # Recurse into children
    if agent.children:
        for child_name, child in agent.children.items():
            _apply_defaults(child, child_name, defaults, global_tools)
