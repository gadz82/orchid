"""Built-in tool + skill configuration models (ADR-017)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .schema_rag import OrchidRAGConfig


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


class OrchidBuiltinToolConfig(BaseModel):
    """A built-in Python tool declared at the YAML top level."""

    handler: str  # dotted import path, e.g. "myproject.tools.dates.format_date"
    description: str = ""
    parameters: dict[str, BuiltinToolParameter] = Field(default_factory=dict)
    inject_to_rag: bool = False  # opt-in: store this tool's results in RAG
    rag_ttl: int | None = None  # per-tool TTL override; None = use agent default
    requires_approval: bool = False  # pause and ask user before executing (HITL)
    # Phase A — parallel tool-call dispatch override for built-in
    # tools.  Built-ins have no MCP annotation to fall back to, so a
    # ``None`` value resolves to ``False`` (sequential).  Set to
    # ``True`` for pure read-only / side-effect-free handlers that
    # the agent may safely batch when ``parallel_tools: true``.
    parallel_safe: bool | None = None
    #: Per-tool RAG override (ADR-024).  When set, this tool's
    #: ingestion / retrieval / namespace / payload-index decisions
    #: use the tool's ``rag`` block instead of the agent's.  ``None``
    #: (the default) means inherit from the agent.  The merge is
    #: shallow per top-level field — tool's ``ingestion`` block wins
    #: only when set; same for ``retrieval``, ``namespace``, etc.
    rag: OrchidRAGConfig | None = None


class OrchidAgentSkillStepConfig(BaseModel):
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
    def _check_step_type(self) -> OrchidAgentSkillStepConfig:
        if self.tool and self.agent:
            raise ValueError("A skill step must set either 'tool' or 'agent', not both")
        if not self.tool and not self.agent:
            raise ValueError("A skill step must set either 'tool' or 'agent'")
        return self


class OrchidAgentSkillConfig(BaseModel):
    """A multi-step workflow within a single agent's domain."""

    description: str = ""
    steps: list[OrchidAgentSkillStepConfig]


class OrchidOrchestratorSkillStepConfig(BaseModel):
    """A single step in an orchestrator-level (cross-agent) skill."""

    agent: str  # agent name to invoke
    instruction: str = ""  # hint passed to the agent via the supervisor


class OrchidOrchestratorSkillConfig(BaseModel):
    """A cross-agent workflow defined at the YAML top level."""

    description: str = ""
    steps: list[OrchidOrchestratorSkillStepConfig]
