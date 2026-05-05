"""
Query transformer registry.

Mirrors :mod:`orchid_ai.rag.strategies` and the proven
:mod:`orchid_ai.agents.strategies` template: register, get-by-name with
fallback, clear for test isolation.

Two transformer scopes are formalised on the ABC via
:attr:`OrchidQueryTransformer.pre_strategy`:

* ``pre_strategy=True`` — runs once at agent entry (e.g. ``reformulate``).
* ``pre_strategy=False`` — runs inside the strategy as a fan-out
  (e.g. ``multi_query``, future ``hyde`` / ``decompose``).

The split is enforced at runtime by
:func:`orchid_ai.core.retrieval.apply_pre_strategy` so a misbehaving
implementation surfaces immediately rather than corrupting the query.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.retrieval import OrchidQueryTransformer
from .decompose import DecomposeTransformer
from .hyde import HyDETransformer
from .multi_query import MultiQueryTransformer
from .reformulate import ReformulateTransformer

logger = logging.getLogger(__name__)


_BUILTINS: dict[str, type[OrchidQueryTransformer]] = {
    "reformulate": ReformulateTransformer,
    "multi_query": MultiQueryTransformer,
    "hyde": HyDETransformer,
    "decompose": DecomposeTransformer,
}

TRANSFORMER_REGISTRY: dict[str, type[OrchidQueryTransformer]] = dict(_BUILTINS)


def register_query_transformer(name: str, cls: type[OrchidQueryTransformer]) -> None:
    """Register a custom query transformer by name.

    Logs a warning when an integrator overwrites a built-in so
    accidental shadowing surfaces during boot.
    """
    if name in TRANSFORMER_REGISTRY and TRANSFORMER_REGISTRY[name] is not cls:
        logger.warning(
            "[QueryTransformers] '%s' already registered (was %s); overwriting with %s",
            name,
            TRANSFORMER_REGISTRY[name].__name__,
            cls.__name__,
        )
    TRANSFORMER_REGISTRY[name] = cls
    logger.info("[QueryTransformers] Registered '%s' → %s", name, cls.__name__)


def clear_query_transformers() -> None:
    """Reset to built-in transformers (useful for test isolation)."""
    TRANSFORMER_REGISTRY.clear()
    TRANSFORMER_REGISTRY.update(_BUILTINS)


def get_query_transformer(name: str, **kwargs: Any) -> OrchidQueryTransformer:
    """Look up and instantiate a query transformer by name.

    Extra keyword arguments are forwarded to the transformer class's
    ``__init__`` — used by the agent runtime to thread per-name
    prompt overrides resolved from
    :class:`OrchidQueryTransformerPromptsConfig`.  Custom transformers
    that don't accept the kwarg should ignore it via a permissive
    ``**kwargs`` parameter or be invoked via :func:`get_query_transformer`
    without prompt overrides.

    Raises :class:`KeyError` for an unknown name — unlike strategy
    resolution, transformers have no safe fallback (a missing
    transformer would silently drop a step the user expected to run).
    """
    cls = TRANSFORMER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown query transformer {name!r}. "
            f"Registered: {sorted(TRANSFORMER_REGISTRY)}. "
            f"Call register_query_transformer({name!r}, cls) before use."
        )
    return cls(**kwargs) if kwargs else cls()


def resolve_transformer_kwargs(name: str, prompts: Any) -> dict[str, Any]:
    """Map a transformer name to constructor kwargs from a prompts config.

    Centralises the "which YAML field maps to which constructor kwarg"
    knowledge so callers (the agent + tests) don't repeat it.  The
    ``prompts`` argument is duck-typed against
    :class:`OrchidQueryTransformerPromptsConfig` — passing ``None``
    yields ``{}`` so callers can short-circuit cleanly.

    The mapping covers only the four built-in transformers; custom
    transformers receive ``{}`` and rely on their own defaults.
    """
    if prompts is None:
        return {}

    if name == "multi_query":
        value = getattr(prompts, "multi_query", None)
        return {"system_prompt": value} if value else {}
    if name == "decompose":
        value = getattr(prompts, "decompose", None)
        return {"system_prompt": value} if value else {}
    if name == "reformulate":
        value = getattr(prompts, "reformulate", None)
        return {"system_prompt": value} if value else {}
    if name == "hyde":
        hyde = getattr(prompts, "hyde", None)
        if hyde is None:
            return {}
        out: dict[str, Any] = {}
        if getattr(hyde, "single", None):
            out["single_prompt"] = hyde.single
        if getattr(hyde, "multi", None):
            out["multi_prompt"] = hyde.multi
        return out
    return {}


__all__ = [
    "DecomposeTransformer",
    "HyDETransformer",
    "MultiQueryTransformer",
    "OrchidQueryTransformer",
    "ReformulateTransformer",
    "TRANSFORMER_REGISTRY",
    "clear_query_transformers",
    "get_query_transformer",
    "register_query_transformer",
    "resolve_transformer_kwargs",
]
