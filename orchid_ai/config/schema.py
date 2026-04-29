"""Backward-compatible re-exports for the agents.yaml configuration schema.

Originally a single 612-line module bundling 19 Pydantic classes plus
the defaults-merging pass; the contracts now live in themed sibling
files. The single import path ``orchid_ai.config.schema`` is unchanged
for every consumer.

Themed modules:
  - :mod:`schema_llm` — LLM configuration
  - :mod:`schema_rag` — RAG settings (defaults + per-agent overrides)
  - :mod:`schema_mcp` — MCP server / tool / auth-mode configs
  - :mod:`schema_skills` — Built-in tools and skills (ADR-017)
  - :mod:`schema_guardrails` — Guardrail rule + chain configs (ADR-018)
  - :mod:`schema_supervisor` — Supervisor + execution-hint configs
  - :mod:`schema_agent` — :class:`OrchidAgentConfig`,
                         :class:`OrchidDefaultsConfig`,
                         :class:`OrchidAgentsConfig`,
                         and the ``_apply_defaults`` recursive merger.
"""

from __future__ import annotations

from .schema_agent import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    _apply_defaults,
)
from .schema_guardrails import OrchidGuardrailRuleConfig, OrchidGuardrailsConfig
from .schema_llm import OrchidLLMConfig
from .schema_mcp import OrchidMCPAuthConfig, OrchidMCPServerConfig, OrchidToolConfig
from .schema_rag import OrchidRAGConfig, OrchidRAGDefaultsConfig
from .schema_skills import (
    BuiltinToolParameter,
    OrchidAgentSkillConfig,
    OrchidAgentSkillStepConfig,
    OrchidBuiltinToolConfig,
    OrchidOrchestratorSkillConfig,
    OrchidOrchestratorSkillStepConfig,
)
from .schema_supervisor import ExecutionHints, OrchidSupervisorConfig

__all__ = [
    "BuiltinToolParameter",
    "ExecutionHints",
    "OrchidAgentConfig",
    "OrchidAgentSkillConfig",
    "OrchidAgentSkillStepConfig",
    "OrchidAgentsConfig",
    "OrchidBuiltinToolConfig",
    "OrchidDefaultsConfig",
    "OrchidGuardrailRuleConfig",
    "OrchidGuardrailsConfig",
    "OrchidLLMConfig",
    "OrchidMCPAuthConfig",
    "OrchidMCPServerConfig",
    "OrchidOrchestratorSkillConfig",
    "OrchidOrchestratorSkillStepConfig",
    "OrchidRAGConfig",
    "OrchidRAGDefaultsConfig",
    "OrchidSupervisorConfig",
    "OrchidToolConfig",
    "_apply_defaults",
]
