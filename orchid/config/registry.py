"""
Agent class registry — maps YAML ``class`` values to Python classes.

Resolution order:
  1. ``None`` → ``GenericAgent`` (default, config-driven)
  2. Short name in registry (e.g. ``"learning"``) → pre-registered class
  3. Dotted import path (e.g. ``"orchid.agents.learning.LearningAgent"``) → dynamic import
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent import BaseAgent

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseAgent]] = {}


def register(name: str, cls: type[BaseAgent]) -> None:
    """Register a custom agent class by short name."""
    _REGISTRY[name] = cls
    logger.debug("[Registry] registered '%s' → %s", name, cls.__name__)


def get_class(class_path: str | None) -> type[BaseAgent]:
    """
    Resolve a class_path to a Python class.

    Parameters
    ----------
    class_path : str | None
        - ``None`` → returns ``GenericAgent``
        - Short name in registry → returns pre-registered class
        - Dotted path (e.g. ``"orchid.agents.learning.LearningAgent"``) → dynamic import

    Returns
    -------
    type[BaseAgent]
        The resolved agent class.
    """
    if class_path is None:
        from ..agents.generic_agent import GenericAgent

        return GenericAgent

    # Try registry first (short names)
    if class_path in _REGISTRY:
        return _REGISTRY[class_path]

    # Dynamic import from dotted path
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        logger.debug("[Registry] dynamically imported '%s' from '%s'", class_name, module_path)
        return cls
    except (ValueError, ImportError, AttributeError) as exc:
        raise ImportError(
            f"Cannot resolve agent class '{class_path}'. "
            f"Ensure it is a valid dotted import path or a registered short name. "
            f"Error: {exc}"
        ) from exc
