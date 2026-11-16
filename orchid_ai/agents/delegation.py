"""Decision-backed external-agent delegation for the ``delegation`` point.

The point answers one bounded question: handle the request **locally** or hand it
to one **configured external CLI agent**. The provider only ever sees a sanitised
catalog — external-agent names and descriptions — never the command, arguments,
working directory, environment, or timeout from the trusted operator YAML. It
returns a name, never a prompt, a command, or tool arguments.

Every uncertain outcome resolves to ``local`` once the point is *active*: an
empty catalog, a malformed allowlist, an unknown or unlisted agent, low
confidence, a refusal, a timeout, a provider error, or an unexpected exception.
An unconfigured/disabled point is different — it keeps the caller's legacy
behaviour. A fallback never starts a subprocess and never removes the
framework's existing local handling.

The helper does not invoke anything. It returns a :class:`DelegationDecision`
that the consuming agent maps onto its existing tool path. Within
``GenericAgent``'s normal run path, ``local`` suppresses the configured
external-agent tools for that run; an accepted external agent is dispatched
through the existing ``ExternalAgentCLITool`` / ``CLIRunner`` path (including its
HITL approval gate), so this module never adds a second subprocess abstraction.
Peer-agent skill steps and mini-agent forks have separate execution paths and
may bypass this gate.

Exactly one structured ``orchid.decision`` line is emitted per provider call.
The line carries no request text, catalog description, command, or secret. An
unconfigured/disabled point emits none; so does an active point that fails closed
before a provider call (missing provider or empty catalog).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..decision.observability import log_decision

if TYPE_CHECKING:
    from ..config.schema_external_agent import OrchidExternalAgentConfig

__all__ = [
    "LOCAL_TARGET",
    "MAX_CANDIDATES",
    "MAX_DESCRIPTION_CHARS",
    "MAX_QUERY_CHARS",
    "DelegationCatalog",
    "DelegationDecision",
    "DelegationRequest",
    "ExternalAgentDescriptor",
    "decide_delegation",
]

logger = logging.getLogger(__name__)

_POINT = "delegation"
_QUESTION = "target"

#: Reserved option that means "handle the request with this agent".
LOCAL_TARGET = "local"

#: Hard bounds for the request state. The query is truncated, descriptions are
#: collapsed and truncated, and the catalog is capped so a hostile or oversized
#: operator config cannot grow the provider prompt without bound.
MAX_QUERY_CHARS = 4000
MAX_DESCRIPTION_CHARS = 240
MAX_CANDIDATES = 32
MAX_AGENT_NAME_CHARS = 128

#: Outer wall-clock ceiling for one advisory delegation decision. Providers also
#: receive this value as their per-call ``timeout``; the outer ``asyncio.timeout``
#: keeps a provider that ignores the parameter from stalling the run.
MAX_DECISION_SECONDS = 2.0

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")

_INSTRUCTIONS = (
    "Choose local to handle the request with this agent. "
    "Choose an external agent only when delegating that specific request clearly "
    "produces a better result and the agent is listed. Choose local when the "
    "request is general, when no external agent clearly fits, or when unsure. "
    "Never choose a name that is not listed."
)


@dataclass(frozen=True)
class ExternalAgentDescriptor:
    """One external agent as the decision provider may see it.

    Only the trusted operator's name and description are exposed. Command paths,
    arguments, environment variables, working directories, timeouts, and
    approval policy never enter the decision boundary.
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("external agent name must be a string")
        if not self.name.strip():
            raise ValueError("external agent name must be non-empty")
        if len(self.name) > MAX_AGENT_NAME_CHARS:
            raise ValueError(f"external agent name exceeds {MAX_AGENT_NAME_CHARS} characters")
        if not isinstance(self.description, str):
            raise TypeError("external agent description must be a string")
        if len(self.description) > MAX_DESCRIPTION_CHARS:
            raise ValueError(f"external agent description exceeds {MAX_DESCRIPTION_CHARS} characters")

    def to_payload(self) -> dict[str, str]:
        """Return the JSON-safe name/description pair sent to the provider."""
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class DelegationCatalog:
    """The bounded external-agent catalog available to one agent.

    The graph builder narrows the fleet-level ``external_agents`` declaration to
    the names the agent is actually allowlisted to call (``agent.tools``).

    An empty catalog has two different meanings, distinguished by the caller:
    when the point is disabled/unconfigured it keeps the legacy tool inventory;
    when the point is active it is a fail-closed local condition — no provider
    call is made and the consuming agent suppresses its external-agent tools.
    """

    agents: tuple[ExternalAgentDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agents, tuple):
            raise TypeError("delegation catalog agents must be a tuple")
        seen: set[str] = set()
        for agent in self.agents:
            if not isinstance(agent, ExternalAgentDescriptor):
                raise TypeError("delegation catalog entries must be ExternalAgentDescriptor values")
            if agent.name in seen:
                raise ValueError(f"duplicate external agent name {agent.name!r}")
            seen.add(agent.name)

    def __bool__(self) -> bool:
        return bool(self.agents)

    @property
    def names(self) -> tuple[str, ...]:
        """Return configured external-agent names in deterministic order."""
        return tuple(agent.name for agent in self.agents)

    @classmethod
    def from_config(
        cls,
        external_agents: Mapping[str, OrchidExternalAgentConfig] | None,
        *,
        allowed_tools: Sequence[str] | None = None,
    ) -> DelegationCatalog:
        """Build a catalog from trusted YAML, narrowed to ``allowed_tools``.

        ``allowed_tools=None`` keeps every configured external agent; the graph
        always passes the agent's ``tools`` list so agent-level allowlists are
        enforced before the provider is asked. Only ``description`` is read from
        each entry — no command, argument, environment, or timeout value.
        """
        allowed = None if allowed_tools is None else {name for name in allowed_tools if isinstance(name, str)}
        descriptors: list[ExternalAgentDescriptor] = []
        for name, config in (external_agents or {}).items():
            if not isinstance(name, str) or not name.strip():
                continue
            if allowed is not None and name not in allowed:
                continue
            if _CONTROL_CHARS_RE.search(name):
                logger.warning("[Delegation] skipping external agent with control characters in its name")
                continue
            if len(name) > MAX_AGENT_NAME_CHARS:
                # An oversized operator key must not abort graph construction:
                # skip it from the decision catalog (it is no longer a valid
                # provider option) and let the caller fail closed locally.
                logger.warning(
                    "[Delegation] skipping external agent name longer than %d characters",
                    MAX_AGENT_NAME_CHARS,
                )
                continue
            raw_description = getattr(config, "description", "")
            description = _bounded_description(raw_description, fallback=name)
            descriptors.append(ExternalAgentDescriptor(name=name, description=description))
        descriptors.sort(key=lambda agent: agent.name)
        return cls(agents=tuple(descriptors))


@dataclass(frozen=True)
class DelegationRequest:
    """Bounded request state for the ``delegation`` point."""

    query: str
    agent_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("delegation query must be a string")
        if len(self.query) > MAX_QUERY_CHARS:
            raise ValueError(f"delegation query exceeds {MAX_QUERY_CHARS} characters")
        if not isinstance(self.agent_name, str):
            raise TypeError("delegation agent_name must be a string")
        if len(self.agent_name) > MAX_AGENT_NAME_CHARS:
            raise ValueError(f"delegation agent_name exceeds {MAX_AGENT_NAME_CHARS} characters")

    @classmethod
    def from_query(cls, query: str, *, agent_name: str = "") -> DelegationRequest:
        """Build a request, truncating the query to the hard bound."""
        if not isinstance(query, str):
            raise TypeError("delegation query must be a string")
        return cls(
            query=query[:MAX_QUERY_CHARS],
            agent_name=_bounded_agent_name(agent_name),
        )


@dataclass(frozen=True)
class DelegationDecision:
    """The mapped outcome of one ``delegation`` decision.

    ``source`` distinguishes a confident provider answer (``decision``) from an
    operational fallback (``fallback``) and an unconfigured/disabled point or an
    empty catalog (``disabled``). ``target`` is :data:`LOCAL_TARGET` whenever the
    request must be handled locally. Only ``decision`` with a non-local target
    delegates.
    """

    target: str
    source: Literal["decision", "fallback", "disabled"]
    confidence: float | None = None
    fallback_reason: str | None = None
    provider: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("delegation target must be a non-empty string")
        if self.source not in ("decision", "fallback", "disabled"):
            raise ValueError(f"unknown delegation source {self.source!r}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("delegation confidence must be within [0, 1]")
        if self.fallback_reason is not None and not isinstance(self.fallback_reason, str):
            raise TypeError("delegation fallback_reason must be a string or None")

    @property
    def accepted(self) -> bool:
        """Whether the provider produced a confident answer (local or external)."""
        return self.source == "decision"

    @property
    def delegated(self) -> bool:
        """Whether the request must be handed to an external agent."""
        return self.source == "decision" and self.target != LOCAL_TARGET


async def decide_delegation(
    binding: OrchidDecisionPointBinding | None,
    request: DelegationRequest,
    catalog: DelegationCatalog,
    *,
    timeout: float | None = MAX_DECISION_SECONDS,
) -> DelegationDecision:
    """Ask the bound ``delegation`` point where the request should run.

    ``binding=None`` means the point is disabled/unconfigured: no provider call,
    no decision line, and the caller keeps its legacy behaviour
    (``source="disabled"``).

    A non-``None`` binding with an empty catalog is different: the point is
    active but has no usable candidate, so the result is a fail-closed local
    decision with no provider call and no decision line. The consuming agent
    then suppresses its external-agent tools rather than falling back to the
    legacy inventory.

    Every operational failure resolves to ``local`` with exactly one structured
    fallback line. The helper never raises for a provider/answer failure, never
    retries, and never includes request text or provider text in its result.
    """
    if not isinstance(request, DelegationRequest):
        raise TypeError("decide_delegation expects a DelegationRequest")
    if not isinstance(catalog, DelegationCatalog):
        raise TypeError("decide_delegation expects a DelegationCatalog")
    if timeout is not None and timeout <= 0:
        raise ValueError("delegation decision timeout must be positive or None")
    if binding is None or not catalog:
        return DelegationDecision(target=LOCAL_TARGET, source="disabled")

    candidates, allowlist_active = _apply_allowlist(binding, catalog.agents)
    if candidates is None:
        return _fallback(
            binding,
            "invalid_options",
            detail="options.agents must list configured external agents",
        )
    if any(agent.name == LOCAL_TARGET for agent in candidates):
        return _fallback(
            binding,
            "invalid_options",
            detail=f"external agent name {LOCAL_TARGET!r} is reserved",
        )
    if not candidates:
        return _fallback(binding, "no_candidates")
    if len(candidates) > MAX_CANDIDATES:
        return _fallback(binding, "too_many_candidates")

    options = (LOCAL_TARGET, *(agent.name for agent in candidates))
    question = OrchidDecisionQuestion(
        name=_QUESTION,
        kind="choice",
        options=options,
        instructions=_INSTRUCTIONS,
    )
    state = OrchidDecisionState(
        text=request.query,
        payload={
            "agent": request.agent_name or None,
            "external_agents": [agent.to_payload() for agent in candidates],
        },
    )

    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            result = await binding.decide(state, [question], timeout=timeout)
    except OrchidDecisionError as exc:
        return _fallback(
            binding,
            exc.code or "provider_error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=exc.provider,
        )
    except TimeoutError:
        return _fallback(
            binding,
            "timeout",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        # Only the exception type crosses this seam: a provider message can
        # contain request text, and a traceback would duplicate it in logs.
        logger.warning("[Delegation] decision provider raised %s", type(exc).__name__)
        return _fallback(
            binding,
            "provider_error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        result_provider = result.provider
        answers = result.answers
    except Exception:
        return _fallback(binding, "invalid_response", latency_ms=latency_ms)

    provider = result_provider if isinstance(result_provider, str) and result_provider else binding.provider
    if not isinstance(answers, Mapping):
        return _fallback(binding, "invalid_response", latency_ms=latency_ms, provider=provider)
    answer = answers.get(_QUESTION)
    if not isinstance(answer, OrchidDecisionAnswer):
        return _fallback(binding, "invalid_response", latency_ms=latency_ms, provider=provider)
    if answer.refusal is not None:
        return _fallback(
            binding,
            "refusal",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.kind != "choice" or not isinstance(answer.value, str):
        return _fallback(
            binding,
            "invalid_response",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )
    if answer.confidence < binding.confidence_threshold:
        return _fallback(
            binding,
            "low_confidence",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    selected = answer.value
    if selected == LOCAL_TARGET:
        log_decision(
            point=_POINT,
            path="decision",
            provider=provider,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            outcome="accepted",
        )
        logger.info("[Delegation] decision selected target=%s", LOCAL_TARGET)
        return DelegationDecision(
            target=LOCAL_TARGET,
            source="decision",
            confidence=answer.confidence,
            provider=provider,
        )

    if selected not in {agent.name for agent in candidates}:
        return _fallback(
            binding,
            "out_of_allowlist" if allowlist_active else "unknown_agent",
            latency_ms=latency_ms,
            confidence=answer.confidence,
            provider=provider,
        )

    log_decision(
        point=_POINT,
        path="decision",
        provider=provider,
        latency_ms=latency_ms,
        confidence=answer.confidence,
        outcome="accepted",
    )
    logger.info("[Delegation] decision selected an external agent")
    return DelegationDecision(
        target=selected,
        source="decision",
        confidence=answer.confidence,
        provider=provider,
    )


def _apply_allowlist(
    binding: OrchidDecisionPointBinding,
    agents: tuple[ExternalAgentDescriptor, ...],
) -> tuple[tuple[ExternalAgentDescriptor, ...] | None, bool]:
    """Narrow the catalog to ``options.agents``, if declared.

    ``None`` signals an invalid declaration (malformed, empty, or containing a
    name that is not configured for this agent). The caller falls back without a
    provider call, so a typo cannot silently shrink or invent the candidate set.
    """
    raw = binding.options.get("agents")
    if raw is None:
        return agents, False
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or not raw:
        return None, False
    if any(not isinstance(name, str) or not name.strip() for name in raw):
        return None, False
    allowed = {name for name in raw}
    configured = {agent.name for agent in agents}
    if any(name not in configured for name in allowed):
        return None, False
    return tuple(agent for agent in agents if agent.name in allowed), True


def _fallback(
    binding: OrchidDecisionPointBinding,
    reason: str,
    *,
    latency_ms: float | None = None,
    confidence: float | None = None,
    provider: str = "",
    detail: str = "",
) -> DelegationDecision:
    """Emit one structured fallback line and keep local handling."""
    message = reason if not detail else f"{reason} ({detail})"
    logger.warning("[Delegation] decision fell back to local handling (reason=%s)", message)
    log_decision(
        point=_POINT,
        path="llm_fallback",
        provider=provider or binding.provider,
        latency_ms=latency_ms,
        confidence=confidence,
        outcome="fallback",
        fallback_reason=reason,
    )
    return DelegationDecision(
        target=LOCAL_TARGET,
        source="fallback",
        confidence=confidence,
        fallback_reason=reason,
        provider=provider or binding.provider,
    )


def _bounded_description(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    text = _CONTROL_CHARS_RE.sub(" ", value)
    text = " ".join(text.split())
    if not text:
        return fallback
    return text[:MAX_DESCRIPTION_CHARS]


def _bounded_agent_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("delegation agent_name must be a string")
    text = _CONTROL_CHARS_RE.sub(" ", value).strip()
    return text[:MAX_AGENT_NAME_CHARS]
