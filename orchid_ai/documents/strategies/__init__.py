"""
Ingestion strategy + chunk post-processor registries (ADR-022).

Mirrors :mod:`orchid_ai.rag.strategies` and
:mod:`orchid_ai.agents.strategies`: register, get-by-name with safe
fallback, clear for tests.

Stage 1 ships the ``recursive`` ingestion strategy (both flat chunking
and parent-in-metadata mode) and no post-processors.  Stage 2 adds
``semantic``, ``hierarchical``, ``headered`` strategies plus the
``contextual_headers`` post-processor.
"""

from __future__ import annotations

import logging

from ...core.ingestion import OrchidChunkPostProcessor, OrchidIngestionStrategy
from ..post_processors.contextual_headers import ContextualHeaderPostProcessor
from .headered import HeaderedIngestion
from .hierarchical import HierarchicalIngestion
from .recursive import RecursiveIngestion
from .semantic import SemanticIngestion

logger = logging.getLogger(__name__)


_INGESTION_BUILTINS: dict[str, type[OrchidIngestionStrategy]] = {
    "recursive": RecursiveIngestion,
    "semantic": SemanticIngestion,
    "hierarchical": HierarchicalIngestion,
    "headered": HeaderedIngestion,
}

_POST_PROCESSOR_BUILTINS: dict[str, type[OrchidChunkPostProcessor]] = {
    "contextual_headers": ContextualHeaderPostProcessor,
}

INGESTION_REGISTRY: dict[str, type[OrchidIngestionStrategy]] = dict(_INGESTION_BUILTINS)
POST_PROCESSOR_REGISTRY: dict[str, type[OrchidChunkPostProcessor]] = dict(_POST_PROCESSOR_BUILTINS)


# ── Ingestion strategies ──────────────────────────────────────


def register_ingestion_strategy(name: str, cls: type[OrchidIngestionStrategy]) -> None:
    """Register a custom ingestion strategy by name."""
    if name in INGESTION_REGISTRY and INGESTION_REGISTRY[name] is not cls:
        logger.warning(
            "[IngestionStrategies] '%s' already registered (was %s); overwriting with %s",
            name,
            INGESTION_REGISTRY[name].__name__,
            cls.__name__,
        )
    INGESTION_REGISTRY[name] = cls
    logger.info("[IngestionStrategies] Registered '%s' → %s", name, cls.__name__)


def clear_ingestion_strategies() -> None:
    """Reset to built-in strategies (useful for test isolation)."""
    INGESTION_REGISTRY.clear()
    INGESTION_REGISTRY.update(_INGESTION_BUILTINS)


def get_ingestion_strategy(name: str) -> OrchidIngestionStrategy:
    """Look up and instantiate an ingestion strategy by name.

    Falls back to ``"recursive"`` when the name is unknown.
    """
    cls = INGESTION_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown ingestion strategy '%s', falling back to 'recursive'", name)
        cls = INGESTION_REGISTRY.get("recursive", RecursiveIngestion)
    return cls()


# ── Chunk post-processors ─────────────────────────────────────


def register_post_processor(name: str, cls: type[OrchidChunkPostProcessor]) -> None:
    """Register a custom chunk post-processor by name."""
    if name in POST_PROCESSOR_REGISTRY and POST_PROCESSOR_REGISTRY[name] is not cls:
        logger.warning(
            "[ChunkPostProcessors] '%s' already registered (was %s); overwriting with %s",
            name,
            POST_PROCESSOR_REGISTRY[name].__name__,
            cls.__name__,
        )
    POST_PROCESSOR_REGISTRY[name] = cls
    logger.info("[ChunkPostProcessors] Registered '%s' → %s", name, cls.__name__)


def clear_post_processors() -> None:
    """Reset to built-in post-processors (useful for test isolation)."""
    POST_PROCESSOR_REGISTRY.clear()
    POST_PROCESSOR_REGISTRY.update(_POST_PROCESSOR_BUILTINS)


def get_post_processor(name: str) -> OrchidChunkPostProcessor:
    """Look up and instantiate a chunk post-processor by name.

    Raises :class:`KeyError` for unknown names — post-processors have no
    safe default (silently dropping a header step would corrupt the
    embedded text).
    """
    cls = POST_PROCESSOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown chunk post-processor {name!r}. "
            f"Registered: {sorted(POST_PROCESSOR_REGISTRY)}. "
            f"Call register_post_processor({name!r}, cls) before use."
        )
    return cls()


__all__ = [
    "ContextualHeaderPostProcessor",
    "HeaderedIngestion",
    "HierarchicalIngestion",
    "INGESTION_REGISTRY",
    "OrchidChunkPostProcessor",
    "OrchidIngestionStrategy",
    "POST_PROCESSOR_REGISTRY",
    "RecursiveIngestion",
    "SemanticIngestion",
    "clear_ingestion_strategies",
    "clear_post_processors",
    "get_ingestion_strategy",
    "get_post_processor",
    "register_ingestion_strategy",
    "register_post_processor",
]
