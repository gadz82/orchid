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
from typing import TYPE_CHECKING

from ...core.ingestion import OrchidChunkPostProcessor, OrchidIngestionStrategy
from ..chunker import ChunkConfig
from ..post_processors.contextual_headers import ContextualHeaderPostProcessor
from ..post_processors.entity_extraction import EntityExtractionPostProcessor
from .headered import HeaderedIngestion
from .hierarchical import HierarchicalIngestion
from .recursive import RecursiveIngestion
from .semantic import SemanticIngestion

if TYPE_CHECKING:
    from ...config.schema_rag import OrchidIngestionConfig

logger = logging.getLogger(__name__)


_INGESTION_BUILTINS: dict[str, type[OrchidIngestionStrategy]] = {
    "recursive": RecursiveIngestion,
    "semantic": SemanticIngestion,
    "hierarchical": HierarchicalIngestion,
    "headered": HeaderedIngestion,
}

_POST_PROCESSOR_BUILTINS: dict[str, type[OrchidChunkPostProcessor]] = {
    "contextual_headers": ContextualHeaderPostProcessor,
    "entity_extraction": EntityExtractionPostProcessor,
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


_CHUNK_CONFIG_STRATEGIES: tuple[type[OrchidIngestionStrategy], ...] = (
    RecursiveIngestion,
    HierarchicalIngestion,
    HeaderedIngestion,
)


def build_ingestion_strategy(config: OrchidIngestionConfig) -> OrchidIngestionStrategy:
    """Construct an ingestion strategy from an :class:`OrchidIngestionConfig`.

    Bridges YAML config → constructed strategy.  For
    :class:`ChunkConfig`-driven strategies (``recursive``,
    ``hierarchical``, ``headered``) the chunk knobs from the YAML
    block flow into ``ChunkConfig``.  Other strategies (e.g.
    ``semantic``) are constructed zero-arg — their per-strategy
    knobs are not (yet) plumbed through ``OrchidIngestionConfig``.

    Falls back to a default-configured ``RecursiveIngestion`` when
    the strategy name is unknown.
    """
    name = config.strategy or "recursive"
    cls = INGESTION_REGISTRY.get(name)
    if cls is None:
        logger.warning(
            "[IngestionStrategies] Unknown strategy '%s' — falling back to 'recursive'",
            name,
        )
        cls = INGESTION_REGISTRY.get("recursive", RecursiveIngestion)

    if issubclass(cls, _CHUNK_CONFIG_STRATEGIES):
        return cls(
            ChunkConfig(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                parent_chunk_size=config.parent_chunk_size,
                parent_chunk_overlap=config.parent_chunk_overlap,
            )
        )
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
    "EntityExtractionPostProcessor",
    "HeaderedIngestion",
    "HierarchicalIngestion",
    "INGESTION_REGISTRY",
    "OrchidChunkPostProcessor",
    "OrchidIngestionStrategy",
    "POST_PROCESSOR_REGISTRY",
    "RecursiveIngestion",
    "SemanticIngestion",
    "build_ingestion_strategy",
    "clear_ingestion_strategies",
    "clear_post_processors",
    "get_ingestion_strategy",
    "get_post_processor",
    "register_ingestion_strategy",
    "register_post_processor",
]
