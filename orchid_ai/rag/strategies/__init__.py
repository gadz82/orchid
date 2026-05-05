"""
Retrieval strategy registry.

Mirrors the proven :mod:`orchid_ai.agents.strategies` template:
register, get-by-name with a safe fallback, clear for test isolation.
Concrete strategies live in this package; integrators register their
own from a composition root via :func:`register_retrieval_strategy`.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.retrieval import OrchidRetrievalStrategy
from .graph_rag import GraphRAGRetrieval
from .hybrid import HybridRetrieval
from .hyde import HyDERetrieval
from .multi_query import MultiQueryRetrieval
from .simple import SimpleRetrieval

logger = logging.getLogger(__name__)


_BUILTINS: dict[str, type[OrchidRetrievalStrategy]] = {
    "simple": SimpleRetrieval,
    "multi_query": MultiQueryRetrieval,
    "hyde": HyDERetrieval,
    "hybrid": HybridRetrieval,
    "graph_rag": GraphRAGRetrieval,
}

RETRIEVAL_REGISTRY: dict[str, type[OrchidRetrievalStrategy]] = dict(_BUILTINS)


def register_retrieval_strategy(name: str, cls: type[OrchidRetrievalStrategy]) -> None:
    """Register a custom retrieval strategy by name.

    Overwrites silently — the most recent registration wins.  Logs a
    warning when an integrator overwrites a built-in so accidental
    shadowing surfaces during boot.
    """
    if name in RETRIEVAL_REGISTRY and RETRIEVAL_REGISTRY[name] is not cls:
        logger.warning(
            "[RetrievalStrategies] '%s' already registered (was %s); overwriting with %s",
            name,
            RETRIEVAL_REGISTRY[name].__name__,
            cls.__name__,
        )
    RETRIEVAL_REGISTRY[name] = cls
    logger.info("[RetrievalStrategies] Registered '%s' → %s", name, cls.__name__)


def clear_retrieval_strategies() -> None:
    """Reset to built-in strategies (useful for test isolation)."""
    RETRIEVAL_REGISTRY.clear()
    RETRIEVAL_REGISTRY.update(_BUILTINS)


def get_retrieval_strategy(
    name: str,
    config: Any = None,
) -> OrchidRetrievalStrategy:
    """Look up and instantiate a retrieval strategy by name.

    When ``config`` is provided (an :class:`OrchidRetrievalConfig`),
    the strategy is built via its :meth:`from_config` classmethod so
    YAML knobs (e.g. ``hyde.n_hypothetical``) flow into construction.
    Without a config the strategy is constructed via the default
    zero-arg ``cls()`` — useful for tests and direct programmatic use.

    Falls back to whatever is registered under ``"simple"`` when the
    name is unknown — mirroring :func:`orchid_ai.agents.strategies.get_strategy`'s
    safe-fallback behaviour so a typo in YAML degrades to today's
    default instead of crashing the agent.
    """
    cls = RETRIEVAL_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown retrieval strategy '%s', falling back to 'simple'", name)
        cls = RETRIEVAL_REGISTRY.get("simple", SimpleRetrieval)
    if config is None:
        return cls()
    return cls.from_config(config)


__all__ = [
    "GraphRAGRetrieval",
    "HybridRetrieval",
    "HyDERetrieval",
    "MultiQueryRetrieval",
    "OrchidRetrievalStrategy",
    "RETRIEVAL_REGISTRY",
    "SimpleRetrieval",
    "clear_retrieval_strategies",
    "get_retrieval_strategy",
    "register_retrieval_strategy",
]
