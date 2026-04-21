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

from ..core.guardrails import OrchidGuardrail, OrchidGuardrailChain

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
    Optional keys: ``fail_action`` (default: "block"), ``config`` (dict
    of guardrail-specific parameters).

    Example YAML structure::

        guardrails:
          input:
            - type: max_length
              fail_action: block
              config:
                max_characters: 5000
            - type: content_safety
              fail_action: block
    """
    chain = OrchidGuardrailChain()

    for cfg in configs:
        type_name = cfg.get("type", "")
        cls = GUARDRAIL_REGISTRY.get(type_name)
        if not cls:
            logger.warning("[Guardrails] Unknown guardrail type '%s' — skipped", type_name)
            continue

        fail_action = cfg.get("fail_action", "block")
        guardrail_config = cfg.get("config", {})

        try:
            instance = cls(fail_action=fail_action, **guardrail_config)
            chain.add(instance)
        except (TypeError, ValueError) as exc:
            logger.error("[Guardrails] Failed to instantiate '%s': %s", type_name, exc)

    return chain


# ── Auto-register built-in guardrails on import ──────────────


def _auto_register() -> None:
    """Import and register all built-in guardrails."""
    from .content_safety import ContentSafetyGuardrail
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


_auto_register()
