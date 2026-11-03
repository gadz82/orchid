"""Decision-model configuration — providers, bindings, budget, and egress.

The blocks are additive and optional. A configuration without
``decision_models``/``decisions`` parses into empty defaults and the framework
behaves exactly as before. Validation happens at load time so a bad provider
reference or an unsafe endpoint fails before a request starts.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DecisionProtocol = Literal["state-questions", "openai-compatible", "llm", "stub"]

_HTTP_PROTOCOLS = frozenset({"state-questions", "openai-compatible"})
_POINTS = ("routing", "skills", "tools", "rag")
_AGENT_POINTS = ("skills", "tools", "rag")

_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "metadata.google.internal"})


def _validate_base_url(value: str, *, provider: str) -> str:
    """Reject endpoints that cannot safely receive user state.

    HTTPS only, no embedded credentials, no loopback/link-local/metadata
    targets, and no query/fragment. A path is allowed because gateways expose
    versioned prefixes.
    """
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError(f"decision provider {provider!r}: base_url must use https")
    if parsed.username or parsed.password:
        raise ValueError(f"decision provider {provider!r}: base_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"decision provider {provider!r}: base_url must not carry a query or fragment")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"decision provider {provider!r}: base_url must include a host")
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise ValueError(f"decision provider {provider!r}: base_url host {hostname!r} is not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return value
    if address.is_loopback or address.is_link_local or address.is_unspecified or address.is_multicast:
        raise ValueError(f"decision provider {provider!r}: base_url host {hostname!r} is not routable")
    return value


def validate_allowed_host(value: str) -> str:
    host = value.strip().lower()
    if not host:
        raise ValueError("allowed_decision_hosts entries must be non-empty")
    if any(marker in host for marker in ("/", ":", "@", "?", "#")):
        raise ValueError(f"allowed_decision_hosts entry {value!r} must be a bare hostname")
    return host


class OrchidDecisionBudgetConfig(BaseModel):
    """Per-turn ceiling and provider breaker settings.

    ``max_calls_per_turn`` bounds how many provider calls one user turn may
    make across all decision points. ``max_total_ms`` bounds the wall clock.
    ``breaker_cooldown_s`` is how long a failing provider is skipped before a
    half-open probe.
    """

    model_config = ConfigDict(extra="forbid")

    max_calls_per_turn: int = Field(default=4, ge=1, le=16)
    max_total_ms: int = Field(default=4000, ge=100, le=60000)
    breaker_cooldown_s: float = Field(default=30.0, ge=0.0, le=600.0)
    failure_threshold: int = Field(default=3, ge=1, le=20)


class OrchidDecisionModelConfig(BaseModel):
    """One declared decision provider.

    ``protocol`` selects the wire dialect, not the vendor. ``systemone`` is
    accepted as an alias for ``state-questions``. ``api_key_env`` names an
    environment variable; the value is never stored in configuration.
    """

    model_config = ConfigDict(extra="forbid")

    protocol: DecisionProtocol = "state-questions"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=2.0, gt=0.0, le=60.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_response_bytes: int = Field(default=262_144, ge=1024, le=1_048_576)
    state_transform: str | None = None

    @field_validator("protocol", mode="before")
    @classmethod
    def _normalise_protocol(cls, value: Any) -> Any:
        if value == "systemone":
            return "state-questions"
        return value

    @model_validator(mode="after")
    def _validate_provider(self) -> OrchidDecisionModelConfig:
        if self.protocol in _HTTP_PROTOCOLS:
            if not self.model.strip():
                raise ValueError(f"decision provider with protocol {self.protocol!r} requires a model")
            if not self.base_url:
                raise ValueError(f"decision provider with protocol {self.protocol!r} requires a base_url")
            _validate_base_url(self.base_url, provider=self.model)
        elif self.protocol == "llm":
            if not self.model.strip():
                raise ValueError("decision provider with protocol 'llm' requires a chat model")
            if self.base_url:
                raise ValueError("decision provider with protocol 'llm' must not set base_url")
        else:
            if self.model.strip():
                raise ValueError("decision provider with protocol 'stub' must not set model")
            if self.base_url:
                raise ValueError("decision provider with protocol 'stub' must not set base_url")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty environment variable name")
        if self.state_transform is not None and not self.state_transform.strip():
            raise ValueError("state_transform must be a non-empty import path")
        return self


class OrchidDecisionPointConfig(BaseModel):
    """A binding from one decision point to a provider.

    ``default`` in :class:`OrchidDecisionsConfig` supplies parameters only; it
    never enables a point on its own. ``escalate`` is valid only for routing,
    because tool and RAG paths have no user-facing seam.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    on_low_confidence: Literal["llm", "default", "escalate"] = "llm"
    max_candidates: int | None = Field(default=None, ge=1, le=64)
    mode: Literal["filter"] | None = None


class OrchidDecisionsConfig(BaseModel):
    """Global decision bindings plus the parameters-only default block."""

    model_config = ConfigDict(extra="forbid")

    budget: OrchidDecisionBudgetConfig = Field(default_factory=OrchidDecisionBudgetConfig)
    default: OrchidDecisionPointConfig | None = None
    routing: OrchidDecisionPointConfig | None = None
    skills: OrchidDecisionPointConfig | None = None
    tools: OrchidDecisionPointConfig | None = None
    rag: OrchidDecisionPointConfig | None = None

    def _global_binding(self, point: str) -> OrchidDecisionPointConfig | None:
        if point not in _POINTS:
            raise ValueError(f"unknown decision point {point!r}")
        return getattr(self, point)

    def resolve(
        self,
        point: str,
        agent: OrchidAgentDecisionsConfig | None = None,
    ) -> OrchidDecisionPointConfig | None:
        """Resolve one point through agent → global → default.

        Returns ``None`` when no block enables the point. ``decisions.default``
        contributes parameters only, so a default block alone never enables a
        decision.
        """
        global_binding = self._global_binding(point)
        agent_binding = agent.get(point) if agent is not None else None
        if agent_binding is None and global_binding is None:
            return None

        merged: dict[str, Any] = {}
        if self.default is not None:
            merged.update(self.default.model_dump(exclude_unset=True))
        if global_binding is not None:
            merged.update(global_binding.model_dump(exclude_unset=True))
        if agent_binding is not None:
            merged.update(agent_binding.model_dump(exclude_unset=True))
        return OrchidDecisionPointConfig.model_validate(merged)


class OrchidAgentDecisionsConfig(BaseModel):
    """Per-agent overrides for the decision points that run inside an agent.

    Routing happens before an agent is known, so it is fleet-level and rejected
    here with an explanatory error rather than a generic extra-field error.
    """

    model_config = ConfigDict(extra="forbid")

    skills: OrchidDecisionPointConfig | None = None
    tools: OrchidDecisionPointConfig | None = None
    rag: OrchidDecisionPointConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_routing(cls, data: Any) -> Any:
        if isinstance(data, dict) and "routing" in data:
            raise ValueError(
                "agents.<name>.decisions.routing is not supported: routing is a fleet-level "
                "decision evaluated before an agent is selected. Configure decisions.routing instead."
            )
        return data

    def get(self, point: str) -> OrchidDecisionPointConfig | None:
        if point not in _AGENT_POINTS:
            raise ValueError(f"decision point {point!r} is not available per agent")
        return getattr(self, point)


def validate_decision_bindings(config: Any) -> None:
    """Validate provider references, egress, and point-specific options.

    Called from the root model validator and again after ``merge_from_db`` so a
    DB-sourced agent cannot bypass it.
    """
    errors: list[str] = []
    enabled: list[str] = []

    for point in _POINTS:
        binding = config.decisions._global_binding(point)
        if binding is None:
            continue
        enabled.append(point)
        _validate_binding_shape(f"decisions.{point}", point, binding, errors)
        resolved = config.decisions.resolve(point)
        model_name = resolved.model if resolved is not None else None
        if not model_name:
            errors.append(f"decisions.{point} is enabled but no provider is selected")
        elif model_name not in config.decision_models:
            errors.append(f"decisions.{point} references unknown decision provider {model_name!r}")

    for agent_name, agent in config.agents.items():
        agent_decisions = getattr(agent, "decisions", None)
        if agent_decisions is None:
            continue
        for point in _AGENT_POINTS:
            binding = agent_decisions.get(point)
            if binding is None:
                continue
            label = f"agents.{agent_name}.decisions.{point}"
            enabled.append(label)
            _validate_binding_shape(label, point, binding, errors)
            resolved = config.decisions.resolve(point, agent_decisions)
            model_name = resolved.model if resolved is not None else None
            if not model_name:
                errors.append(f"{label} is enabled but no provider is selected")
            elif model_name not in config.decision_models:
                errors.append(f"{label} references unknown decision provider {model_name!r}")

    if enabled and not config.allowed_decision_hosts:
        errors.append(
            "allowed_decision_hosts must list at least one host when a decision point is enabled "
            f"(enabled: {', '.join(enabled)})"
        )

    if errors:
        raise ValueError("; ".join(errors))


def _validate_binding_shape(
    label: str,
    point: str,
    binding: OrchidDecisionPointConfig,
    errors: list[str],
) -> None:
    if binding.on_low_confidence == "escalate" and point != "routing":
        errors.append(f"{label}.on_low_confidence 'escalate' is only valid for routing")
    if binding.max_candidates is not None and point != "tools":
        errors.append(f"{label}.max_candidates is only valid for tools")
    if binding.mode is not None and point != "rag":
        errors.append(f"{label}.mode is only valid for rag")
