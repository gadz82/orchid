"""Registry of declarative decision points.

The registry is the single place that knows *which* decision points exist. It
holds :class:`~orchid_ai.core.decision.OrchidDecisionPointSpec` values only:
name, scope, candidate ceiling, fallback declaration, and an optional
point-specific validator. Flow behaviour (payload shaping, result mapping,
fallbacks) belongs to the adapter that consumes the point binding.

The four legacy points — ``routing``, ``skills``, ``tools``, ``rag`` — and the
nine extensible flow points — ``retrieval_strategy``, ``model_routing``,
``tool_choice``, ``guardrails``, ``memory``, ``events``, ``ingestion``,
``delegation``, ``agentic_loop`` — are registered as built-ins on import so
configuration validation and provider composition can be registry-driven
without changing their YAML keys or behaviour. Consumers register new points
with :func:`register_decision_point` *before* loading a configuration that
references them.

Import cost: importing this module initialises the ``orchid_ai.decision``
package, which pulls in the factory and adapters. Code that only needs the
dependency-free contracts should import :mod:`orchid_ai.core.decision` instead.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from typing import Any

from ..core.decision import (
    OrchidDecisionFallbackPolicy,
    OrchidDecisionPointScope,
    OrchidDecisionPointSpec,
)

__all__ = [
    "DECISION_POINT_REGISTRY",
    "get_decision_point",
    "iter_decision_points",
    "list_decision_points",
    "register_decision_point",
]

logger = logging.getLogger(__name__)

#: Point name → specification. Populated with the built-ins at import time.
DECISION_POINT_REGISTRY: dict[str, OrchidDecisionPointSpec] = {}


def register_decision_point(
    spec: OrchidDecisionPointSpec,
    *,
    override: bool = False,
) -> OrchidDecisionPointSpec:
    """Register ``spec`` and return it.

    Duplicate names are rejected. Built-in points can never be replaced; a
    consumer point may be re-registered only with ``override=True``, which keeps
    reload-style integrations working without silently shadowing the framework.
    """
    if not isinstance(spec, OrchidDecisionPointSpec):
        raise TypeError("register_decision_point expects an OrchidDecisionPointSpec")
    existing = DECISION_POINT_REGISTRY.get(spec.name)
    if existing is not None:
        if existing.builtin:
            raise ValueError(f"decision point {spec.name!r} is a built-in and cannot be replaced")
        if not override:
            raise ValueError(
                f"decision point {spec.name!r} is already registered; pass override=True to replace a consumer point"
            )
        logger.warning("[Decision] Replacing registered decision point %r", spec.name)
    DECISION_POINT_REGISTRY[spec.name] = spec
    return spec


def get_decision_point(name: str) -> OrchidDecisionPointSpec:
    """Return the spec registered under ``name``.

    Raises :class:`KeyError` listing every registered point so a YAML typo is
    actionable at load time.
    """
    try:
        return DECISION_POINT_REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(DECISION_POINT_REGISTRY)) or "(none)"
        raise KeyError(f"unknown decision point {name!r}; registered points: {registered}") from None


def list_decision_points(scope: OrchidDecisionPointScope | None = None) -> tuple[str, ...]:
    """Return registered point names sorted alphabetically, optionally by scope."""
    _validate_scope(scope)
    return tuple(sorted(name for name, spec in DECISION_POINT_REGISTRY.items() if scope is None or spec.scope == scope))


def iter_decision_points(scope: OrchidDecisionPointScope | None = None) -> Iterator[OrchidDecisionPointSpec]:
    """Yield registered specs in deterministic (name-sorted) order."""
    for name in list_decision_points(scope):
        yield DECISION_POINT_REGISTRY[name]


def _validate_scope(scope: str | None) -> None:
    if scope is not None and scope not in ("fleet", "agent"):
        raise ValueError(f"decision point scope must be 'fleet' or 'agent', got {scope!r}")


def _register_builtins() -> None:
    """Register the four legacy points with their unchanged behaviour contract.

    The fallback modes describe the existing behaviour; they do not activate a
    provider. A configuration without a binding still resolves to ``None``.
    """
    builtins = (
        OrchidDecisionPointSpec(
            name="routing",
            scope="fleet",
            max_candidates=32,
            fallback="llm",
            description="Choose the agents or orchestrator skill for a request.",
            builtin=True,
        ),
        OrchidDecisionPointSpec(
            name="skills",
            scope="agent",
            fallback="llm",
            description="Choose the agent skill that handles a request.",
            builtin=True,
        ),
        OrchidDecisionPointSpec(
            name="tools",
            scope="agent",
            max_candidates=64,
            fallback="all_tools",
            description="Shortlist the tools bound for one turn.",
            builtin=True,
        ),
        OrchidDecisionPointSpec(
            name="rag",
            scope="agent",
            fallback="keep_original",
            description="Gate retrieved passages before they reach the prompt.",
            builtin=True,
        ),
    )
    for spec in builtins:
        register_decision_point(spec)


def _register_extended_points() -> None:
    """Register the nine extensible flow points with their fallback contracts.

    Scope and candidate limits describe where the adapter evaluates the point
    and how many candidates it may send; they carry no flow logic. Fallback
    modes are stable codes the consuming adapter implements, and every fallback
    preserves the framework's existing behaviour (configured strategy, default
    model, deterministic processors, no delegation, the round cap).
    """
    extended = (
        OrchidDecisionPointSpec(
            name="retrieval_strategy",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="configured_strategy",
                description="Use the retrieval strategy configured on the agent.",
            ),
            description="Choose the retrieval strategy for one query.",
            builtin=True,
            validate_config=_validate_retrieval_strategy_options,
        ),
        OrchidDecisionPointSpec(
            name="model_routing",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="default_model",
                description="Use the agent or fleet default chat model.",
            ),
            description="Choose the chat model or fallback for a request.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="tool_choice",
            scope="agent",
            max_candidates=32,
            fallback=OrchidDecisionFallbackPolicy(
                mode="all_tools",
                description="Bind every available tool (the existing all strategy).",
            ),
            description="Choose and order the tools bound for one turn.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="guardrails",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="configured_guardrails",
                description="Run the configured guardrail chain and its per-rail failure policy.",
            ),
            description="Choose allow, block, redact, or warn for input and output content.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="memory",
            scope="fleet",
            fallback=OrchidDecisionFallbackPolicy(
                mode="configured_memory",
                description="Keep the configured memory strategy and sliding-window heuristic.",
            ),
            description="Choose when to summarise, keep, or persist conversation memory.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="events",
            scope="fleet",
            fallback=OrchidDecisionFallbackPolicy(
                mode="deterministic_processors",
                description="Run the deterministic event processors without dropping or duplicating the event.",
            ),
            description="Classify, deduplicate, and activate framework events.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="ingestion",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="registry_default",
                description="Use the registry default parser, chunker, and strategy for the document.",
            ),
            description="Choose the ingestion strategy for a document.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="delegation",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="no_delegation",
                description="Handle the request locally without invoking an external agent.",
            ),
            description="Choose whether and where to delegate a request to an external agent.",
            builtin=True,
            validate_config=_validate_options,
        ),
        OrchidDecisionPointSpec(
            name="agentic_loop",
            scope="agent",
            fallback=OrchidDecisionFallbackPolicy(
                mode="max_tool_rounds",
                description="Continue until the agent's configured max_tool_rounds cap.",
            ),
            description="Choose whether the agentic loop continues or stops after a round.",
            builtin=True,
            validate_config=_validate_options,
        ),
    )
    for spec in extended:
        register_decision_point(spec)


_MAX_OPTION_DEPTH = 10


def _validate_options(binding: Mapping[str, Any]) -> None:
    """Reject point options a JSON transport cannot carry.

    Options are point-specific and owned by the adapter, so this shared
    validator deliberately checks only the transport contract: string keys,
    bounded nesting, and finite scalars. A non-JSON value would otherwise fail
    when the adapter builds its request state.
    """
    options = binding.get("options")
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("options must be a mapping")  # noqa: TRY004 — validators are contracted to raise ValueError
    _validate_json_safe(options, path="options", depth=0)


def _validate_retrieval_strategy_options(binding: Mapping[str, Any]) -> None:
    """Validate the optional ``options.strategies`` candidate allowlist.

    The allowlist narrows the registry-driven candidate set; it never adds a
    strategy that is not registered. A name that is unknown at config-load time
    fails here, so a typo cannot silently shrink the prompt.
    """
    _validate_options(binding)
    options = binding.get("options") or {}
    strategies = options.get("strategies")
    if strategies is None:
        return
    if (
        not isinstance(strategies, list)
        or not strategies
        or any(not isinstance(name, str) or not name.strip() for name in strategies)
    ):
        raise ValueError("options.strategies must be a non-empty list of strategy names")

    from ..rag.strategies import RETRIEVAL_REGISTRY

    unknown = sorted(name for name in strategies if name not in RETRIEVAL_REGISTRY)
    if unknown:
        raise ValueError(f"options.strategies contains unknown retrieval strategies: {', '.join(unknown)}")


def _validate_json_safe(value: Any, *, path: str, depth: int) -> None:
    if depth > _MAX_OPTION_DEPTH:
        raise ValueError(f"{path} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_safe(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            _validate_json_safe(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} contains a non-JSON value ({type(value).__name__})")


_register_builtins()
_register_extended_points()
