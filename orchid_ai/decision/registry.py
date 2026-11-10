"""Registry of declarative decision points.

The registry is the single place that knows *which* decision points exist. It
holds :class:`~orchid_ai.core.decision.OrchidDecisionPointSpec` values only:
name, scope, candidate ceiling, fallback declaration, and an optional
point-specific validator. Flow behaviour (payload shaping, result mapping,
fallbacks) belongs to the adapter that consumes the point binding.

The four legacy points — ``routing``, ``skills``, ``tools``, ``rag`` — are
registered as built-ins on import so configuration validation and provider
composition can be registry-driven without changing their YAML keys or
behaviour. Consumers register new points with :func:`register_decision_point`
*before* loading a configuration that references them.

Import cost: importing this module initialises the ``orchid_ai.decision``
package, which pulls in the factory and adapters. Code that only needs the
dependency-free contracts should import :mod:`orchid_ai.core.decision` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from ..core.decision import OrchidDecisionPointScope, OrchidDecisionPointSpec

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


_register_builtins()
