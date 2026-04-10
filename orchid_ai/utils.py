"""
General-purpose utilities shared across the orchid framework.
"""

from __future__ import annotations

import importlib


def import_class(class_path: str) -> type:
    """
    Dynamically import a class by its dotted path.

    Parameters
    ----------
    class_path : str
        Fully-qualified dotted path, e.g. ``"orchid_ai.persistence.postgres.PostgresChatStorage"``.

    Raises
    ------
    ImportError
        If the module or attribute cannot be found.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ImportError(f"Cannot resolve class '{class_path}': {exc}") from exc
