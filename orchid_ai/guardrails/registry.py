"""
OrchidGuardrail registry — maps type names to concrete guardrail classes.

Follows the same Open/Closed pattern as ``STRATEGY_REGISTRY`` in
``agents/strategies.py``.  New guardrails are added via
``register_guardrail()`` — no code changes required.

The ``build_guardrail_chain()`` factory constructs a ``OrchidGuardrailChain``
from YAML configuration dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailChain,
    OrchidGuardrailFailurePolicyAware,
)

logger = logging.getLogger(__name__)

# ── Registry ─────────────────────────────────────────────────

GUARDRAIL_REGISTRY: dict[str, type[OrchidGuardrail]] = {}


def register_guardrail(name: str, cls: type[OrchidGuardrail]) -> None:
    """Register a guardrail class under the given type name."""
    GUARDRAIL_REGISTRY[name] = cls
    logger.debug("[Guardrails] Registered '%s' → %s", name, cls.__name__)


def get_guardrail(name: str) -> type[OrchidGuardrail] | None:
    """Look up a guardrail class by type name."""
    return GUARDRAIL_REGISTRY.get(name)


# ── Factory ──────────────────────────────────────────────────


def build_guardrail_chain(
    configs: list[dict[str, Any]],
) -> OrchidGuardrailChain:
    """
    Build a ``OrchidGuardrailChain`` from a list of YAML config dicts.

    Each dict must have a ``type`` key matching a registered guardrail.
    An unknown type raises an actionable ``ValueError`` at build time; config
    load defers type resolution so startup hooks can register custom rails.
    Optional keys: ``fail_action`` (default: "block"), ``on_failure``
    (required for rails that declare
    :class:`~orchid_ai.core.guardrails.OrchidGuardrailFailurePolicyAware`,
    with ``open`` or ``closed``), and ``config`` (dict of guardrail-specific
    parameters).

    Only failure-aware rails receive ``on_failure``. Deterministic rails keep
    their existing constructors and defaults; a policy declared on one of them
    is warned about and ignored.

    Example YAML structure::

        guardrails:
          input:
            - type: max_length
              fail_action: block
              config:
                max_characters: 5000
            - type: content_safety
              fail_action: block
            - type: decision
              fail_action: block
              on_failure: closed
    """
    chain = OrchidGuardrailChain()

    for cfg in configs:
        type_name = cfg.get("type", "")
        cls = GUARDRAIL_REGISTRY.get(type_name)
        if not cls:
            registered = ", ".join(sorted(GUARDRAIL_REGISTRY)) or "(none)"
            raise ValueError(f"unknown guardrail type {type_name!r}; registered types: {registered}")

        fail_action = cfg.get("fail_action", "block")
        guardrail_config = cfg.get("config", {})
        if not isinstance(guardrail_config, dict):
            logger.error("[Guardrails] Guardrail '%s' config must be a mapping — skipped", type_name)
            continue
        guardrail_config = dict(guardrail_config)
        if "on_failure" in guardrail_config:
            raise ValueError(
                f"guardrail type '{type_name}': on_failure must be declared at the rule level, not under config"
            )

        on_failure = cfg.get("on_failure")
        if issubclass(cls, OrchidGuardrailFailurePolicyAware):
            if on_failure is None:
                raise ValueError(f"guardrail type '{type_name}' requires an explicit on_failure ('open' or 'closed')")
            guardrail_config["on_failure"] = on_failure
        elif on_failure is not None:
            logger.warning(
                "[Guardrails] on_failure is ignored for deterministic guardrail type '%s'",
                type_name,
            )

        try:
            instance = cls(fail_action=fail_action, **guardrail_config)
            chain.add(instance)
        except (TypeError, ValueError) as exc:
            if issubclass(cls, OrchidGuardrailFailurePolicyAware):
                # A misconfigured failable rail must fail loudly, not silently
                # disable a safety check.
                raise ValueError(  # noqa: TRY004 — builder config errors are ValueError
                    f"invalid guardrail config for '{type_name}': {exc}"
                ) from exc
            logger.error("[Guardrails] Failed to instantiate '%s': %s", type_name, exc)

    return chain


# ── Auto-register built-in guardrails on import ──────────────


def _auto_register() -> None:
    """Import and register all built-in guardrails."""
    from .content_safety import ContentSafetyGuardrail
    from .decision_guardrail import DecisionGuardrail
    from .groundedness import GroundednessGuardrail
    from .max_length import MaxLengthGuardrail
    from .pii import PIIDetectionGuardrail
    from .prompt_injection import PromptInjectionGuardrail
    from .topic_restriction import TopicRestrictionGuardrail

    register_guardrail("content_safety", ContentSafetyGuardrail)
    register_guardrail("prompt_injection", PromptInjectionGuardrail)
    register_guardrail("pii_detection", PIIDetectionGuardrail)
    register_guardrail("topic_restriction", TopicRestrictionGuardrail)
    register_guardrail("max_length", MaxLengthGuardrail)
    register_guardrail("groundedness", GroundednessGuardrail)
    register_guardrail("decision", DecisionGuardrail)


_auto_register()
