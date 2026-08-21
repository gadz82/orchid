"""
General-purpose utilities shared across the orchid framework.
"""

from __future__ import annotations

import importlib


def unwrap_exception_group(exc: BaseException) -> BaseException:
    """Return the first concrete exception inside an ``ExceptionGroup``.

    ``asyncio.TaskGroup`` and ``ExceptionGroup`` produce summary strings
    like "unhandled errors in a TaskGroup (1 sub-exception)" which hide
    the real failure.  This helper walks down the first exception in each
    group until it reaches the actual leaf exception, suitable for logging
    and error reporting.
    """
    while isinstance(exc, BaseExceptionGroup):
        if not exc.exceptions:
            break
        exc = exc.exceptions[0]
    return exc


def import_class(class_path: str) -> type:
    """
    Dynamically import a class by its dotted path.

    Parameters
    ----------
    class_path : str
        Fully-qualified dotted path, e.g. ``"orchid_storage_postgres.OrchidPostgresChatStorage"``.

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
