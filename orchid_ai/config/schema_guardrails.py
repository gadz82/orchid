"""Guardrail configuration models (ADR-018)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OrchidGuardrailRuleConfig(BaseModel):
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


class OrchidGuardrailsConfig(BaseModel):
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

    input: list[OrchidGuardrailRuleConfig] = Field(default_factory=list)
    output: list[OrchidGuardrailRuleConfig] = Field(default_factory=list)
