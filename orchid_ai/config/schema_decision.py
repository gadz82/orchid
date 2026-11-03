"""Decision-model configuration — providers, bindings, budget, and egress.

The blocks are additive and optional. A configuration without
``decision_models``/``decisions`` parses into empty defaults and the framework
behaves exactly as before. Validation happens at load time so a bad provider
reference, an unsafe endpoint, or an allowlist mismatch fails before a request
starts.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import OrchidConfigError

_PROTOCOL_ALIASES = {"systemone": "state-questions", "openai-decisions": "openai-compatible"}
_HTTP_LIKE_EXCLUDED = frozenset({"llm", "stub"})
_AGENT_POINTS = ("skills", "tools", "rag")
_ALL_POINTS = ("routing", "skills", "tools", "rag")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata.goog",
    }
)
_BLOCKED_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud metadata
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDSv2 IPv6
    }
)


def normalise_host(value: str, *, label: str) -> str:
    """Canonicalise a hostname or IP literal for comparison and safety checks."""
    host = value.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        raise ValueError(f"{label}: host must be non-empty")
    if any(marker in host for marker in ("/", "*", "@", "?", "#", " ", "\\")):
        raise ValueError(f"{label}: host {value!r} contains an invalid character")
    if host.count(":") == 1:
        raise ValueError(f"{label}: host {value!r} must not include a port")
    host = host.rstrip(".")
    if not host or host.startswith(".") or ".." in host:
        raise ValueError(f"{label}: host {value!r} is not a valid name")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"{label}: host {value!r} is not a valid international name") from exc
    return host


_NAT64_NETWORK = ipaddress.ip_network("64:ff9b::/96")


def _parse_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return the address for a literal host, or ``None`` for a DNS name.

    IPv4-mapped and NAT64 IPv6 forms are unwrapped so a loopback or metadata
    target cannot hide behind an IPv6 literal.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        return mapped
    if isinstance(address, ipaddress.IPv6Address) and address in _NAT64_NETWORK:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def _reject_numeric_alias(host: str, *, label: str) -> None:
    """Reject non-canonical numeric hosts that resolvers expand to loopback.

    ``127.1``, ``2130706433``, and ``0x7f.0.0.1`` all resolve to ``127.0.0.1``
    through ``getaddrinfo`` but are not parseable by :mod:`ipaddress`.
    """
    compact = host.replace(".", "")
    numeric = compact.isdigit() or host.startswith("0x")
    if numeric and _parse_address(host) is None:
        raise ValueError(f"{label}: host {host!r} is a non-canonical numeric address")


def _validate_host_safety(host: str, *, label: str) -> None:
    if host in _BLOCKED_HOSTNAMES:
        raise ValueError(f"{label}: host {host!r} is not allowed")
    address = _parse_address(host)
    if address is None:
        return
    if address in _BLOCKED_ADDRESSES:
        raise ValueError(f"{label}: host {host!r} is a metadata endpoint")
    if address.is_loopback or address.is_link_local or address.is_unspecified or address.is_multicast:
        raise ValueError(f"{label}: host {host!r} is not routable")


def validate_decision_base_url(value: str, *, provider: str) -> str:
    """Reject endpoints that cannot safely receive user state.

    HTTPS only, no embedded credentials, no loopback/link-local/metadata
    targets, and no query/fragment. A path is allowed because gateways expose
    versioned prefixes. DNS aliases to metadata endpoints cannot be detected
    here; the transport resolves and validates every address before a request.
    """
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError(f"decision provider {provider!r}: base_url must use https")
    if "@" in parsed.netloc:
        raise ValueError(f"decision provider {provider!r}: base_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"decision provider {provider!r}: base_url must not carry a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"decision provider {provider!r}: base_url has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"decision provider {provider!r}: base_url port must be between 1 and 65535")
    if not parsed.hostname:
        raise ValueError(f"decision provider {provider!r}: base_url must include a host")
    host = normalise_host(parsed.hostname, label=f"decision provider {provider!r}")
    _reject_numeric_alias(host, label=f"decision provider {provider!r}")
    _validate_host_safety(host, label=f"decision provider {provider!r}")
    return value


def validate_resolved_address(value: str, *, label: str = "decision provider") -> None:
    """Reject a resolved peer address that the egress policy forbids."""
    address = _parse_address(value)
    if address is None:
        raise ValueError(f"{label}: resolved peer {value!r} is not an IP address")
    if address in _BLOCKED_ADDRESSES:
        raise ValueError(f"{label}: resolved peer {value!r} is a metadata endpoint")
    if address.is_loopback or address.is_link_local or address.is_unspecified or address.is_multicast:
        raise ValueError(f"{label}: resolved peer {value!r} is not routable")


def validate_allowed_host(value: str) -> str:
    """Normalise and safety-check one ``allowed_decision_hosts`` entry."""
    host = normalise_host(value, label="allowed_decision_hosts")
    _reject_numeric_alias(host, label="allowed_decision_hosts")
    _validate_host_safety(host, label="allowed_decision_hosts")
    return host


class OrchidDecisionBudgetConfig(BaseModel):
    """Per-turn ceiling and provider breaker settings.

    ``max_calls_per_turn`` bounds how many provider calls one user turn may
    make across all decision points. ``max_total_ms`` bounds the wall clock.
    ``breaker_cooldown_s`` is how long a failing provider is skipped before a
    half-open probe. ``failure_threshold`` is the number of consecutive
    failures that opens the breaker.
    """

    model_config = ConfigDict(extra="forbid")

    max_calls_per_turn: int = Field(default=4, ge=1, le=16)
    max_total_ms: int = Field(default=4000, ge=100, le=60000)
    breaker_cooldown_s: float = Field(default=30.0, ge=0.0, le=600.0)
    failure_threshold: int = Field(default=3, ge=1, le=20)


class OrchidStubDecisionConfig(BaseModel):
    """Deterministic answers for the ``stub`` protocol.

    Keys are question names or ``"<state text>:<question>"`` contextual keys.
    Values are spec mappings accepted by :class:`StubDecisionModel`.
    """

    model_config = ConfigDict(extra="forbid")

    answers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    mode: Literal["answer", "low_confidence", "fail"] = "answer"


class OrchidDecisionModelConfig(BaseModel):
    """One declared decision provider.

    # ORCHID-MEM: 01a1a11f-f53f-78ee-8b4c-6a1d186786f7 — Declared decision provider with protocol, endpoint, and egress validation.

    ``protocol`` selects the wire dialect, not the vendor. Aliases
    ``systemone`` and ``openai-decisions`` normalise to their neutral names.
    ``api_key_env`` names an environment variable and is validated as an
    identifier so a pasted secret is rejected at load time.
    """

    model_config = ConfigDict(extra="forbid")

    protocol: str = "state-questions"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    decide_path: str | None = None
    timeout_seconds: float = Field(default=2.0, gt=0.0, le=60.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_response_bytes: int = Field(default=262_144, ge=1024, le=1_048_576)
    state_transform: str | None = None
    stub: OrchidStubDecisionConfig | None = None

    @field_validator("protocol", mode="before")
    @classmethod
    def _normalise_protocol(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalised = value.strip().lower()
        return _PROTOCOL_ALIASES.get(normalised, normalised)

    @model_validator(mode="after")
    def _validate_provider(self) -> OrchidDecisionModelConfig:
        if self.protocol == "stub":
            if self.model.strip():
                raise ValueError("decision provider with protocol 'stub' must not set model")
            if self.base_url:
                raise ValueError("decision provider with protocol 'stub' must not set base_url")
        elif self.protocol == "llm":
            if not self.model.strip():
                raise ValueError("decision provider with protocol 'llm' requires a chat model")
            if self.base_url:
                raise ValueError("decision provider with protocol 'llm' must not set base_url")
        else:
            # HTTP dialects, including custom protocols registered at runtime.
            if not self.model.strip():
                raise ValueError(f"decision provider with protocol {self.protocol!r} requires a model")
            if not self.base_url:
                raise ValueError(f"decision provider with protocol {self.protocol!r} requires a base_url")
            validate_decision_base_url(self.base_url, provider=self.model)
        if self.api_key_env is not None:
            env_name = self.api_key_env.strip()
            if not _ENV_NAME_RE.match(env_name):
                raise ValueError(
                    "api_key_env must be an environment variable identifier "
                    "(letters, digits, underscore; not a pasted secret)"
                )
            self.api_key_env = env_name
        if self.state_transform is not None and not self.state_transform.strip():
            raise ValueError("state_transform must be a non-empty import path")
        if self.decide_path is not None and (not self.decide_path.startswith("/") or not self.decide_path.strip()):
            raise ValueError("decide_path must be an absolute path starting with '/'")
        if self.stub is not None and self.protocol != "stub":
            raise ValueError("the stub block is only valid with protocol 'stub'")
        return self


class OrchidDecisionPointConfig(BaseModel):
    """A binding from one decision point to a provider.

    ``enabled=False`` disables the point for the scope it is declared on,
    including an inherited global binding. ``escalate`` is valid only for
    routing, because tool and RAG paths have no user-facing seam.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    model: str | None = None
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    on_low_confidence: Literal["llm", "default", "escalate"] = "llm"
    max_candidates: int | None = Field(default=None, ge=1, le=64)
    mode: Literal["filter"] | None = None


class OrchidDecisionDefaultConfig(BaseModel):
    """Parameters inherited by every enabled point. Never enables a point.

    ``model``, ``max_candidates``, and ``mode`` are deliberately absent: they
    are binding or point-specific and a fleet-wide default would make them
    ambiguous.
    """

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    on_low_confidence: Literal["llm", "default"] = "llm"


class OrchidDecisionsConfig(BaseModel):
    """Global decision bindings plus the parameters-only default block."""

    model_config = ConfigDict(extra="forbid")

    budget: OrchidDecisionBudgetConfig = Field(default_factory=OrchidDecisionBudgetConfig)
    default: OrchidDecisionDefaultConfig | None = None
    routing: OrchidDecisionPointConfig | None = None
    skills: OrchidDecisionPointConfig | None = None
    tools: OrchidDecisionPointConfig | None = None
    rag: OrchidDecisionPointConfig | None = None

    def _global_binding(self, point: str) -> OrchidDecisionPointConfig | None:
        if point not in _ALL_POINTS:
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
        decision. An ``enabled=False`` override disables an inherited binding;
        ``routing`` is fleet-level and ignores per-agent overrides.
        """
        global_binding = self._global_binding(point)
        agent_binding = agent.get(point) if agent is not None and point in _AGENT_POINTS else None

        if agent_binding is not None and agent_binding.enabled is False:
            return None
        if agent_binding is None and global_binding is not None and global_binding.enabled is False:
            return None
        if agent_binding is None and global_binding is None:
            return None

        merged: dict[str, Any] = {}
        if self.default is not None:
            merged.update(self.default.model_dump(exclude_unset=True))
        if global_binding is not None and global_binding.enabled is not False:
            merged.update(global_binding.model_dump(exclude_unset=True))
        if agent_binding is not None:
            merged.update(agent_binding.model_dump(exclude_unset=True))
        merged.pop("enabled", None)
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


def iter_agents(agents: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    for name, agent in agents.items():
        path = f"{prefix}.{name}" if prefix else name
        found.append((path, agent))
        children = getattr(agent, "children", None)
        if children:
            found.extend(iter_agents(children, path))
    return found


def validate_decision_bindings(config: Any) -> None:
    """Validate provider references, egress, and point-specific options.

    Called from the root model validator and again after ``merge_from_db`` so a
    DB-sourced agent cannot bypass it. Nested agents are validated with a path
    label. An allowlisted host must match the provider's base URL at load time;
    the transport repeats the check before every request.
    """
    errors: list[str] = []
    enabled: list[str] = []
    http_enabled: list[str] = []

    for point in _ALL_POINTS:
        binding = config.decisions._global_binding(point)
        if binding is None:
            continue
        resolved = config.decisions.resolve(point)
        if resolved is None:
            continue
        label = f"decisions.{point}"
        enabled.append(label)
        _validate_binding_shape(label, point, resolved, errors)
        _validate_provider_reference(config, label, resolved, errors, expected_transform=None)
        if _uses_http(config, resolved):
            http_enabled.append(label)

    for path, agent in iter_agents(config.agents):
        agent_model = getattr(agent, "decision_model", None)
        if agent_model and agent_model not in config.decision_models:
            errors.append(f"agents.{path}.decision_model references unknown decision provider {agent_model!r}")
        agent_decisions = getattr(agent, "decisions", None)
        if agent_decisions is None:
            continue
        for point in _AGENT_POINTS:
            binding = agent_decisions.get(point)
            if binding is None:
                continue
            resolved = config.decisions.resolve(point, agent_decisions)
            if resolved is None:
                continue
            label = f"agents.{path}.decisions.{point}"
            enabled.append(label)
            _validate_binding_shape(label, point, resolved, errors)
            expected_transform = _provider_transform(config, config.decisions.resolve(point))
            _validate_provider_reference(config, label, resolved, errors, expected_transform=expected_transform)
            if _uses_http(config, resolved):
                http_enabled.append(label)

    if http_enabled and not config.allowed_decision_hosts:
        errors.append(
            "allowed_decision_hosts must list at least one host when an HTTP decision point is enabled "
            f"(enabled: {', '.join(http_enabled)})"
        )

    if errors:
        raise OrchidConfigError("; ".join(errors))


def _uses_http(config: Any, binding: OrchidDecisionPointConfig) -> bool:
    if not binding.model:
        return False
    provider = config.decision_models.get(binding.model)
    return provider is not None and provider.protocol not in _HTTP_LIKE_EXCLUDED


def _provider_transform(config: Any, binding: OrchidDecisionPointConfig | None) -> str | None:
    if binding is None or not binding.model:
        return None
    provider = config.decision_models.get(binding.model)
    return provider.state_transform if provider is not None else None


def _validate_provider_reference(
    config: Any,
    label: str,
    resolved: OrchidDecisionPointConfig,
    errors: list[str],
    *,
    expected_transform: str | None,
) -> None:
    model_name = resolved.model
    if not model_name:
        errors.append(f"{label} is enabled but no provider is selected")
        return
    provider = config.decision_models.get(model_name)
    if provider is None:
        errors.append(f"{label} references unknown decision provider {model_name!r}")
        return
    if expected_transform is not None and provider.state_transform != expected_transform:
        errors.append(
            f"{label} selects provider {model_name!r} which changes the global state_transform; "
            "per-agent overrides may not weaken redaction"
        )
    if provider.protocol in _HTTP_LIKE_EXCLUDED:
        return
    assert provider.base_url is not None  # validated by OrchidDecisionModelConfig
    host = normalise_host(urlsplit(provider.base_url).hostname or "", label=label)
    if host not in config.allowed_decision_hosts:
        errors.append(
            f"{label} uses provider {model_name!r} with host {host!r}, which is not listed in allowed_decision_hosts"
        )


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
