"""
Pydantic v2 models for the agents.yaml configuration schema (ADR-016, ADR-017, ADR-018).

The ``class`` YAML key is mapped to ``class_path`` to avoid the Python
reserved word.  A model validator on ``AgentConfig`` merges in default
values from the parent ``AgentsConfig.defaults``.

ADR-017 additions: BuiltinToolConfig, AgentSkillStepConfig, AgentSkillConfig,
OrchestratorSkillStepConfig, OrchestratorSkillConfig.

ADR-018 additions: GuardrailRuleConfig, GuardrailsConfig for global and
per-agent input/output guardrails.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Leaf configs ──────────────────────────────────────────────


class LLMConfig(BaseModel):
    """LLM settings — can appear at defaults or agent level.

    The model string uses LiteLLM's ``provider/model-name`` format:
      - ``gemini/gemini-2.5-flash``          → Google AI Studio
      - ``groq/llama-3.3-70b-versatile``     → Groq
      - ``anthropic/claude-sonnet-4-20250514``  → Anthropic
      - ``ollama/llama3.2``                  → Local Ollama
      - ``openai/gpt-4o``                   → OpenAI

    The optional ``fallback_model`` is tried automatically when the
    primary model fails (503, rate limit, timeout).  When set at
    ``defaults.llm`` level, it applies to all agents and the supervisor
    unless overridden per-agent or per-supervisor.

    Example YAML::

        defaults:
          llm:
            model: gemini/gemini-2.5-flash
            fallback_model: ollama/llama3.2

        agents:
          critical-agent:
            llm:
              model: openai/gpt-4o
              fallback_model: anthropic/claude-sonnet-4-20250514
    """

    model: str = "gemini/gemini-2.5-flash"
    temperature: float = 0.2
    fallback_model: str | None = None


class RAGDefaultsConfig(BaseModel):
    """Default RAG settings inherited by all agents."""

    k: int = 5
    enabled: bool = True
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)
    reformulate_queries: bool = True  # rewrite queries using conversation history
    retriever_type: Literal["simple", "multi_query"] = "simple"


class RAGConfig(BaseModel):
    """Per-agent RAG settings.

    ``retriever_type`` controls the retrieval strategy:
    - ``simple`` (default) — single vector similarity search per query.
    - ``multi_query`` — LLM generates query variations for broader
      recall, then results are merged and deduplicated.

    When ``retriever_type`` is ``None`` (the YAML default), the value
    is inherited from ``defaults.rag.retriever_type``.  Set it
    explicitly to override the default.
    """

    namespace: str = ""
    k: int = 5
    enabled: bool = True
    reformulate_queries: bool = True  # rewrite queries using conversation history
    retriever_type: Literal["simple", "multi_query"] | None = None  # None = inherit from defaults
    rag_ttl: int = 0  # seconds; 0 = no cache (always call tools)


class ToolConfig(BaseModel):
    """A single MCP tool available to an agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    inject_to_rag: bool = False  # opt-in: store this tool's results in RAG
    rag_ttl: int | None = None  # per-tool TTL override; None = use agent default
    requires_approval: bool = False  # pause and ask user before executing (HITL)


class MCPAuthConfig(BaseModel):
    """Per-server authentication configuration.

    Determines how the MCP client authenticates with the server:

    - ``none`` (default): no authentication headers are sent.  Suitable
      for local MCP servers or remote servers that do not require auth.
    - ``passthrough``: forwards the graph's ``AuthContext`` bearer token
      unchanged — the MCP server shares the same identity provider.
    - ``oauth``: the server requires its own OAuth 2.0 flow with a
      third-party identity provider.  Tokens are stored per-user,
      per-server in the ``MCPTokenStore``.

    For ``oauth`` mode, provide either ``issuer`` (OIDC auto-discovery)
    or explicit ``authorization_endpoint`` + ``token_endpoint``.

    Example YAML::

        auth:
          mode: oauth
          client_id: orchid-crm-integration
          issuer: https://auth.crm-provider.com
          scopes: "openid crm.read crm.write"
    """

    mode: Literal["none", "passthrough", "oauth"] = "none"
    client_id: str = ""
    authorization_endpoint: str = ""  # explicit, OR use issuer for OIDC discovery
    token_endpoint: str = ""
    scopes: str = "openid"
    issuer: str = ""  # OIDC auto-discovery endpoint


class MCPServerConfig(BaseModel):
    """An MCP server connected to an agent.

    ``tools``, ``prompts``, and ``resources`` each accept either:
      - An explicit allow-list (e.g. ``["list_types", "create_notification"]``)
      - The wildcard ``"*"`` to discover ALL capabilities from the server

    When any of them is ``"*"``, the corresponding ``discover_all_*`` flag is
    set and the allow-list is cleared — the agent will call ``list_tools()``,
    ``list_prompts()``, or ``list_resources()`` at runtime.
    """

    name: str
    type: Literal["local", "remote"] = "local"
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    url: str  # supports ${ENV_VAR} interpolation (resolved by loader)
    auth: MCPAuthConfig = Field(default_factory=MCPAuthConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)  # prompt names to load (or "*")
    resources: list[str] = Field(default_factory=list)  # resource URIs/names to load (or "*")
    tool_call_strategy: Literal["all", "sequential", "llm_decides"] = "all"
    cache_ttl: int = 300  # seconds — capabilities cache lifetime (0 = no cache)
    discover_all_tools: bool = False
    discover_all_prompts: bool = False
    discover_all_resources: bool = False

    @property
    def discover_all(self) -> bool:
        """True when ALL capabilities (tools + prompts + resources) are wildcarded."""
        return self.discover_all_tools and self.discover_all_prompts and self.discover_all_resources

    @model_validator(mode="before")
    @classmethod
    def _handle_wildcards(cls, data: Any) -> Any:
        """Convert ``"*"`` / ``["*"]`` into the corresponding ``discover_all_*`` flags."""
        if not isinstance(data, dict):
            return data

        for field, flag in [
            ("tools", "discover_all_tools"),
            ("prompts", "discover_all_prompts"),
            ("resources", "discover_all_resources"),
        ]:
            value = data.get(field)
            if value == "*" or value == ["*"]:
                data[field] = []
                data[flag] = True

        return data


class ExecutionHints(BaseModel):
    """Hints for the Supervisor when routing."""

    parallel_safe: bool = True


# ── Built-in tools + Skills (ADR-017) ────────────────────────


class BuiltinToolParameter(BaseModel):
    """A single parameter of a built-in tool.

    Declared in YAML to document the tool's interface for LLM-based
    invocation and Claude Code skill generation.  When omitted in YAML,
    parameters are auto-extracted from the Python function signature
    at registration time.

    Example YAML::

        parameters:
          player_name:
            type: string
            description: "Full or partial player name"
            required: false
            default: ""
    """

    type: str = "string"  # string, int, float, bool
    description: str = ""
    required: bool = True
    default: Any = None


class BuiltinToolConfig(BaseModel):
    """A built-in Python tool declared at the YAML top level."""

    handler: str  # dotted import path, e.g. "myproject.tools.dates.format_date"
    description: str = ""
    parameters: dict[str, BuiltinToolParameter] = Field(default_factory=dict)
    inject_to_rag: bool = False  # opt-in: store this tool's results in RAG
    rag_ttl: int | None = None  # per-tool TTL override; None = use agent default
    requires_approval: bool = False  # pause and ask user before executing (HITL)


class AgentSkillStepConfig(BaseModel):
    """A single step in an agent-level skill.

    A step is either a **tool call** (``tool`` + ``source``) or an
    **agent invocation** (``agent`` + ``instruction``).  Exactly one
    of ``tool`` or ``agent`` must be set.

    Agent invocation lets a skill call another agent directly without
    passing through the Supervisor.  The invoked agent runs its full
    pipeline (RAG + MCP + LLM) and its results chain forward via
    ``previous_results`` like any other step.
    """

    # ── Tool call fields (existing) ──
    tool: str | None = None  # tool name (MCP tool or built-in tool)
    source: str | None = None  # MCP server name, "builtin", or None (= builtin)
    arguments: dict[str, Any] = Field(default_factory=dict)

    # ── Agent invocation fields (new) ──
    agent: str | None = None  # name of another agent to invoke
    instruction: str = ""  # query/instruction sent to the invoked agent

    @property
    def step_key(self) -> str:
        """Canonical key for this step (tool name or agent name)."""
        return self.tool or self.agent or "unknown"

    @model_validator(mode="after")
    def _check_step_type(self) -> AgentSkillStepConfig:
        if self.tool and self.agent:
            raise ValueError("A skill step must set either 'tool' or 'agent', not both")
        if not self.tool and not self.agent:
            raise ValueError("A skill step must set either 'tool' or 'agent'")
        return self


class AgentSkillConfig(BaseModel):
    """A multi-step workflow within a single agent's domain."""

    description: str = ""
    steps: list[AgentSkillStepConfig]


class OrchestratorSkillStepConfig(BaseModel):
    """A single step in an orchestrator-level (cross-agent) skill."""

    agent: str  # agent name to invoke
    instruction: str = ""  # hint passed to the agent via the supervisor


class OrchestratorSkillConfig(BaseModel):
    """A cross-agent workflow defined at the YAML top level."""

    description: str = ""
    steps: list[OrchestratorSkillStepConfig]


# ── Guardrails config (ADR-018) ─────────────────────────────


class GuardrailRuleConfig(BaseModel):
    """A single guardrail rule declaration.

    Maps to a registered guardrail type in the guardrail registry.
    The ``config`` dict is passed as keyword arguments to the guardrail
    constructor.

    Example YAML::

        - type: content_safety
          fail_action: block
          config:
            categories: [self_harm, violence]
        - type: pii_detection
          fail_action: redact
          config:
            entities: [email, phone, ssn]
    """

    type: str  # registered guardrail type name (e.g. "content_safety")
    fail_action: Literal["block", "warn", "redact", "log"] = "block"
    config: dict[str, Any] = Field(default_factory=dict)


class GuardrailsConfig(BaseModel):
    """Input and output guardrail chains.

    Used at both the global level (``orchid.yml``) and per-agent level
    (``agents.yaml``).  Global guardrails run on every request; per-agent
    guardrails run only when that agent is active.

    Example YAML::

        guardrails:
          input:
            - type: prompt_injection
              fail_action: block
            - type: max_length
              fail_action: block
              config:
                max_characters: 10000
          output:
            - type: pii_detection
              fail_action: redact
              config:
                entities: [email, ssn]
    """

    input: list[GuardrailRuleConfig] = Field(default_factory=list)
    output: list[GuardrailRuleConfig] = Field(default_factory=list)


# ── Supervisor config ────────────────────────────────────────


class SupervisorConfig(BaseModel):
    """Supervisor prompt and behavior configuration.

    Allows consumers to customize the assistant name, prompts, and
    conversation history limits without modifying library code.
    When prompt fields are ``None``, the default templates in
    ``supervisor.py`` are used.

    History settings control how much prior conversation context is
    passed to the LLM during routing, synthesis, and sequential
    handoff steps:

    - ``history_max_turns``: maximum user/assistant exchange pairs
      to include.  Default 20 (= up to 40 messages).
    - ``history_max_chars``: maximum characters per individual
      message before truncation.  Default 1000.  Truncated messages
      get an ``…`` suffix.
    """

    assistant_name: str = "AI assistant"
    fallback_model: str | None = None  # fallback LLM for supervisor (overrides defaults.llm.fallback_model)
    streaming_enabled: bool = True  # enable SSE streaming for responses (default: on)
    routing_system_prompt: str | None = None
    synthesis_system_prompt: str | None = None
    sequential_advance_prompt: str | None = None
    history_max_turns: int = 20
    history_max_chars: int = 1000

    # ── Sliding-window summarization (context compression) ──
    history_summary_enabled: bool = True
    history_summary_model: str | None = None  # None = use the supervisor model
    history_summary_recent_turns: int = 10  # keep last N turns verbatim


# ── Agent config (recursive for nesting) ─────────────────────


class AgentConfig(BaseModel):
    """
    Configuration for a single agent.

    When ``class_path`` is ``None``, the ``GenericAgent`` is used.
    When set, it must be either:
      - A dotted Python import path (e.g. ``src.agents.learning.LearningAgent``)
      - A short name registered in the agent class registry
    """

    # Set programmatically by the loader (= the YAML dict key)
    name: str = ""

    description: str
    prompt: str

    # ``class`` in YAML → ``class_path`` in Python (reserved word)
    class_path: str | None = Field(default=None, alias="class")

    rag: RAGConfig = Field(default_factory=RAGConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    llm: LLMConfig | None = None
    execution_hints: ExecutionHints = Field(default_factory=ExecutionHints)

    # Built-in tools available to this agent (ADR-017)
    tools: list[str] = Field(default_factory=list)

    # Agent-level skills — multi-step workflows within this agent (ADR-017)
    skills: dict[str, AgentSkillConfig] = Field(default_factory=dict)

    # Per-agent guardrails (ADR-018)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)

    # Recursive nesting — sub-agents under this agent
    children: dict[str, AgentConfig] | None = None

    # Computed at validation — tool names whose results are injected to RAG
    injectable_tools: set[str] = Field(default_factory=set, exclude=True)

    # Computed at validation — tool name → effective TTL (seconds) for RAG cache
    # Only includes tools with inject_to_rag=True AND effective TTL > 0
    injectable_tool_ttls: dict[str, int] = Field(default_factory=dict, exclude=True)

    # Computed at validation — tool names that require human approval (HITL)
    approval_tools: set[str] = Field(default_factory=set, exclude=True)

    model_config = {"populate_by_name": True}


# ── Defaults config ──────────────────────────────────────────


class DefaultsConfig(BaseModel):
    """Top-level defaults inherited by every agent."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGDefaultsConfig = Field(default_factory=RAGDefaultsConfig)


# ── Root config ──────────────────────────────────────────────


class AgentsConfig(BaseModel):
    """
    Root configuration loaded from agents.yaml.

    After validation, each ``AgentConfig`` has its defaults merged in
    and its ``name`` set from the YAML dict key.
    """

    version: str = "1"
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    # Global built-in tool declarations (ADR-017)
    tools: dict[str, BuiltinToolConfig] = Field(default_factory=dict)

    # Orchestrator-level skills — cross-agent workflows (ADR-017)
    skills: dict[str, OrchestratorSkillConfig] = Field(default_factory=dict)

    # Supervisor configuration — prompt customization
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)

    # Global guardrails — applied to all requests (ADR-018)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)

    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_defaults_and_names(self) -> AgentsConfig:
        """Merge defaults into each agent and set names recursively."""
        for agent_name, agent in self.agents.items():
            _apply_defaults(agent, agent_name, self.defaults, self.tools)
        return self


def _apply_defaults(
    agent: AgentConfig,
    name: str,
    defaults: DefaultsConfig,
    global_tools: dict[str, BuiltinToolConfig] | None = None,
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

    # Collect injectable MCP tool names + TTLs
    agent_ttl = agent.rag.rag_ttl
    for server in agent.mcp_servers:
        for tool in server.tools:
            if isinstance(tool, ToolConfig) and tool.inject_to_rag:
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
            if isinstance(tool, ToolConfig) and tool.requires_approval:
                agent.approval_tools.add(tool.name)
    if global_tools:
        for tool_name in agent.tools:
            tool_cfg = global_tools.get(tool_name)
            if tool_cfg and tool_cfg.requires_approval:
                agent.approval_tools.add(tool_name)

    # Recurse into children
    if agent.children:
        for child_name, child in agent.children.items():
            _apply_defaults(child, child_name, defaults, global_tools)
