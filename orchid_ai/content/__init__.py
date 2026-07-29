from __future__ import annotations

import logging
from typing import Any

from ..core.content import OrchidContentSource
from .local import LocalFileContentSource

logger = logging.getLogger(__name__)

CONTENT_SOURCE_REGISTRY: dict[str, type[OrchidContentSource]] = {}


def register_content_source(name: str, cls: type[OrchidContentSource]) -> None:
    if name in CONTENT_SOURCE_REGISTRY and CONTENT_SOURCE_REGISTRY[name] is not cls:
        logger.warning("[ContentSources] '%s' already registered; overwriting", name)
    CONTENT_SOURCE_REGISTRY[name] = cls
    logger.debug("[ContentSources] Registered '%s'", name)


def build_content_source(name: str, **settings: Any) -> OrchidContentSource:
    cls = CONTENT_SOURCE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown content source {name!r}. "
            f"Registered: {sorted(CONTENT_SOURCE_REGISTRY)}. "
            f"Call register_content_source({name!r}, cls) "
            f"before constructing Orchid."
        )
    return cls(**settings)


register_content_source("local", LocalFileContentSource)
