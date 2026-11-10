"""Narrow per-point bindings for graph consumers.

A binding is the only decision-layer object a flow adapter should receive. It
exposes the selected provider key, the point's threshold, candidate limit and
read-only options, a ``decide`` method with the existing provider contract, and
safe metadata for graph state. Concrete providers, transports and config models
stay behind it.

Resolution is registry-driven: only points declared in YAML (legacy keys or the
additive ``decisions.points`` map) produce a binding, so an unconfigured point —
and every consumer point by default — has no binding and no provider call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..config.schema_decision import OrchidDecisionPointConfig
from ..core.decision import (
    OrchidDecisionModel,
    OrchidDecisionPointBinding,
    OrchidDecisionPointSpec,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from ..decision.observability import safe_log_token
from ..decision.registry import get_decision_point, iter_decision_points

if TYPE_CHECKING:
    from ..runtime import OrchidRuntime

__all__ = [
    "DecisionBinding",
    "bind_decision_points",
    "resolve_decision_binding",
]

logger = logging.getLogger(__name__)


class DecisionBinding:
    """Concrete :class:`~orchid_ai.core.decision.OrchidDecisionPointBinding`.

    The resolved config block and the provider instance are private: consumers
    use the protocol properties and ``decide``. ``options`` is an immutable copy,
    so a consumer cannot mutate the shared configuration object.
    """

    __slots__ = ("_config", "_model", "_spec")

    def __init__(
        self,
        *,
        spec: OrchidDecisionPointSpec,
        config: OrchidDecisionPointConfig,
        model: OrchidDecisionModel,
    ) -> None:
        self._spec = spec
        self._config = config
        self._model = model

    @property
    def spec(self) -> OrchidDecisionPointSpec:
        return self._spec

    @property
    def point(self) -> str:
        return self._spec.name

    @property
    def provider(self) -> str:
        return self._config.model or ""

    @property
    def confidence_threshold(self) -> float:
        return self._config.confidence_threshold

    @property
    def max_candidates(self) -> int | None:
        """Effective candidate ceiling, clamped to the spec's hard limit."""
        configured = self._config.max_candidates
        declared = self._spec.max_candidates
        if configured is None:
            return declared
        if declared is None:
            return configured
        return min(configured, declared)

    @property
    def options(self) -> Mapping[str, Any]:
        """Immutable view of the point-specific YAML options."""
        return MappingProxyType(dict(self._config.options))

    @property
    def fallback_policy(self) -> str:
        return self._spec.fallback.mode

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Sequence[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        return await self._model.decide(state, questions, timeout=timeout)

    def metadata(self) -> Mapping[str, Any]:
        """Structured, payload-free fields safe for graph state and logs.

        Token values go through :func:`~orchid_ai.decision.observability.safe_log_token`
        so an operator-supplied provider key cannot inject a newline or a fake
        ``field=value`` pair into downstream logs.
        """
        return {
            "decision_point": safe_log_token(self.point),
            "provider": safe_log_token(self.provider),
            "confidence_threshold": self.confidence_threshold,
            "max_candidates": self.max_candidates,
            "on_low_confidence": safe_log_token(self._config.on_low_confidence),
            "fallback": safe_log_token(self.fallback_policy),
        }


def resolve_decision_binding(
    config: Any,
    runtime: OrchidRuntime,
    point: str,
    *,
    agent_name: str | None = None,
) -> OrchidDecisionPointBinding | None:
    """Resolve one point for a fleet or agent scope.

    ``None`` has two operational meanings: the point is disabled/unconfigured,
    or it is configured but its provider was not built at the composition root.
    Consumers fall back identically in both cases; the second case additionally
    logs a warning so a missing provider is diagnosable.

    Passing ``agent_name`` for a fleet-scoped point is a programmer error and
    raises ``ValueError`` — fleet points have no per-agent override by design.
    """
    spec = get_decision_point(point)
    if agent_name is not None and spec.scope != "agent":
        raise ValueError(
            f"decision point {point!r} is {spec.scope}-scoped and cannot be resolved for agent {agent_name!r}"
        )

    binding_config = config.resolve_decision(point, agent_name)
    if binding_config is None or not binding_config.model:
        return None
    provider = runtime.get_decision_model(binding_config.model)
    if provider is None:
        logger.warning(
            "[Decision] point %r references provider %r, which is missing from the runtime",
            point,
            binding_config.model,
        )
        return None
    return DecisionBinding(spec=spec, config=binding_config, model=provider)


def bind_decision_points(config: Any, runtime: OrchidRuntime) -> dict[str, OrchidDecisionPointBinding]:
    """Resolve every enabled global point and expose it on ``runtime``.

    Called after runtime assembly (bootstrap) and after a hot reload rebuilds
    providers, so a disabled point cannot leave a stale binding behind. The map
    is keyed by point name; per-agent bindings are resolved on demand with
    :func:`resolve_decision_binding`.
    """
    bindings: dict[str, OrchidDecisionPointBinding] = {}
    runtime.decision_bindings.clear()
    for spec in iter_decision_points():
        binding = resolve_decision_binding(config, runtime, spec.name)
        if binding is None:
            continue
        runtime.register_decision_binding(binding)
        bindings[spec.name] = binding
    return bindings
