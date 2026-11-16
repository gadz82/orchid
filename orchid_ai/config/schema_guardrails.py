"""Guardrail configuration models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Keys the wiring owns. Declaring one under ``config`` would be silently
#: overridden (or duplicate a constructor argument), so config load rejects it.
_RESERVED_CONFIG_KEYS = frozenset(
    {
        "type",
        "fail_action",
        "on_failure",
        "binding",
        "point",
        "configuration_reason",
    }
)


class OrchidGuardrailRuleConfig(BaseModel):
    """A single guardrail rule declaration.

    Maps to a registered guardrail type in the guardrail registry.
    The ``config`` dict is passed as keyword arguments to the guardrail
    constructor; unknown keys are rejected so a typo cannot silently disable a
    rail. ``type`` may name a custom guardrail registered by a startup hook
    after config load; the type is resolved and validated at graph build.

    ADR-002 posture: a decision-backed rail must state ``on_failure``
    explicitly (there is no implicit fail-open for ``type: decision``).
    Deterministic rails never fail and must not declare it.

    Example YAML::

        - type: content_safety
          fail_action: block
          config:
            categories: [self_harm, violence]
        - type: pii_detection
          fail_action: redact
          config:
            entities: [email, phone, ssn]
        - type: decision
          fail_action: block
          on_failure: closed
    """

    model_config = ConfigDict(extra="forbid")

    type: str  # registered guardrail type name (e.g. "content_safety")
    fail_action: Literal["block", "warn", "redact", "log"] = "block"
    #: Per-rail failure posture. ``open`` lets content pass when a rail cannot
    #: obtain a verdict; ``closed`` applies ``fail_action`` instead. Required
    #: for rails that declare ``OrchidGuardrailFailurePolicyAware`` (currently
    #: ``type: decision``). Deterministic rails reject it.
    on_failure: Literal["open", "closed"] | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_failure_policy_and_config(self) -> OrchidGuardrailRuleConfig:
        from ..guardrails.registry import GUARDRAIL_REGISTRY

        # A custom guardrail may be registered by a startup hook *after* config
        # load, so an unknown type is deferred to graph build. Known types are
        # validated here; ``build_guardrail_chain`` raises for anything still
        # unregistered when the chain is actually built.
        if self.type in GUARDRAIL_REGISTRY:
            failable = _is_failure_policy_rail(self.type)

            if failable and self.on_failure is None:
                raise ValueError(
                    f"guardrail type {self.type!r} requires an explicit on_failure "
                    "('open' or 'closed'); ADR-002 has no implicit safety posture"
                )
            if not failable and self.on_failure is not None:
                raise ValueError(
                    f"on_failure is not supported for guardrail type {self.type!r}: "
                    "only rails that declare OrchidGuardrailFailurePolicyAware can fail"
                )

        reserved = sorted(_RESERVED_CONFIG_KEYS & set(self.config))
        if reserved:
            raise ValueError(
                f"guardrail config for {self.type!r} must not declare reserved keys: {', '.join(reserved)}"
            )

        if self.type == "decision":
            _validate_decision_config(self)
        return self


def _is_failure_policy_rail(type_name: str) -> bool:
    """Return whether the registered class owns a runtime failure policy."""
    from ..core.guardrails import OrchidGuardrailFailurePolicyAware
    from ..guardrails.registry import get_guardrail

    cls = get_guardrail(type_name)
    return cls is not None and issubclass(cls, OrchidGuardrailFailurePolicyAware)


def _validate_decision_config(rule: OrchidGuardrailRuleConfig) -> None:
    """Validate the decision rail's constructor arguments at config load.

    The graph builder re-raises for failure-aware rails, but a typo must fail
    here instead of disabling the rail during graph construction.
    """
    from ..guardrails.decision_guardrail import DecisionGuardrail

    try:
        DecisionGuardrail(
            binding=None,
            fail_action=rule.fail_action,
            on_failure=rule.on_failure,
            **rule.config,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid 'decision' guardrail config: {exc}") from exc


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
            - type: decision
              fail_action: block
              on_failure: closed
          output:
            - type: pii_detection
              fail_action: redact
              config:
                entities: [email, ssn]
    """

    model_config = ConfigDict(extra="forbid")

    input: list[OrchidGuardrailRuleConfig] = Field(default_factory=list)
    output: list[OrchidGuardrailRuleConfig] = Field(default_factory=list)
