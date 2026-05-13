"""Umbrella module that re-exports the agents.yaml configuration schema.

The Pydantic classes live in themed sibling files; this module is the
single import path (``orchid_ai.config.schema``) every consumer uses.

Themed modules:
  - :mod:`schema_llm` — LLM configuration
  - :mod:`schema_rag` — RAG settings (defaults + per-agent overrides)
  - :mod:`schema_mcp` — MCP server / tool / auth-mode configs
  - :mod:`schema_skills` — Built-in tools and skills
  - :mod:`schema_guardrails` — Guardrail rule + chain configs
  - :mod:`schema_supervisor` — Supervisor + execution-hint configs
  - :mod:`schema_agent` — :class:`OrchidAgentConfig`,
                         :class:`OrchidDefaultsConfig`,
                         :class:`OrchidAgentsConfig`,
                         and the recursive defaults merger.
"""

from __future__ import annotations

from .schema_agent import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
)
from .schema_events import (
    ActAsUserIdentity,
    AddressedToUserIdentity,
    OrchidEventsConfig,
    OrchidEventsIngestionConfig,
    OrchidIngestionSourceConfig,
    OrchidProcessorConfig,
    OrchidQueueConfig,
    OrchidScheduleConfig,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    OrchidTriggerRetryConfig,
    OrchidValidatorConfig,
    ServiceAccountIdentity,
)
from .schema_guardrails import OrchidGuardrailRuleConfig, OrchidGuardrailsConfig
from .schema_llm import OrchidLLMConfig
from .schema_mcp import OrchidMCPAuthConfig, OrchidMCPServerConfig, OrchidToolConfig
from .schema_mini_agent import OrchidMiniAgentConfig
from .schema_prompts import (
    OrchidAgentPromptConfig,
    OrchidHydeTransformerPromptsConfig,
    OrchidQueryTransformerPromptsConfig,
)
from .schema_rag import (
    OrchidIngestionConfig,
    OrchidRAGConfig,
    OrchidRAGDefaultsConfig,
    OrchidRetrievalConfig,
)
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
    "ActAsUserIdentity",
    "AddressedToUserIdentity",
    "OrchidAgentConfig",
    "OrchidAgentPromptConfig",
    "OrchidAgentSkillConfig",
    "OrchidAgentSkillStepConfig",
    "OrchidAgentsConfig",
    "OrchidBuiltinToolConfig",
    "OrchidDefaultsConfig",
    "OrchidEventsConfig",
    "OrchidEventsIngestionConfig",
    "OrchidGuardrailRuleConfig",
    "OrchidGuardrailsConfig",
    "OrchidHydeTransformerPromptsConfig",
    "OrchidIngestionConfig",
    "OrchidIngestionSourceConfig",
    "OrchidLLMConfig",
    "OrchidMCPAuthConfig",
    "OrchidMCPServerConfig",
    "OrchidMiniAgentConfig",
    "OrchidOrchestratorSkillConfig",
    "OrchidOrchestratorSkillStepConfig",
    "OrchidProcessorConfig",
    "OrchidQueryTransformerPromptsConfig",
    "OrchidQueueConfig",
    "OrchidRAGConfig",
    "OrchidRAGDefaultsConfig",
    "OrchidRetrievalConfig",
    "OrchidScheduleConfig",
    "OrchidSupervisorConfig",
    "OrchidToolConfig",
    "OrchidTriggerConfig",
    "OrchidTriggerEmitConfig",
    "OrchidTriggerMatchConfig",
    "OrchidTriggerRetryConfig",
    "OrchidValidatorConfig",
    "ServiceAccountIdentity",
]
