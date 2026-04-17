"""
Entry-point plugin discovery — shared by orchid-api, orchid-cli, and
anything else that wants to load optional hooks from installed packages.

The function :func:`iter_entry_point_plugins` walks a named entry-point
group and yields ``(name, loaded_object)`` pairs, isolating the caller
from the minor API differences between Python 3.9–3.11 and 3.12+.

Example::

    from orchid_ai.plugins import iter_entry_point_plugins

    for name, router in iter_entry_point_plugins("orchid_api.routers", logger=log):
        if isinstance(router, APIRouter):
            app.include_router(router)

Design notes
------------
* Individual plugin load failures log a warning and are skipped — one
  broken third-party package must not block the whole startup.
* If the stdlib ``importlib.metadata`` module itself is missing (very
  old Python), the generator yields nothing.  This is the only exception
  type the helper swallows — every other error is surfaced via
  ``logger.warning`` so real bugs remain visible.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

__all__ = ["iter_entry_point_plugins"]


def iter_entry_point_plugins(
    group: str,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, loaded_object)`` for every plugin in the entry-point *group*.

    Plugins that fail to load (missing deps, import errors, bad
    entry-point) are logged and skipped.  The caller inspects the
    loaded object and decides whether it's suitable (e.g. "is this a
    ``Typer`` app?", "is this an ``APIRouter``?").

    Parameters
    ----------
    group
        The entry-point group name (e.g. ``"orchid_api.routers"``).
    logger
        Optional logger used for warnings.  When ``None``, a
        module-level logger is used.
    """
    log = logger or logging.getLogger(__name__)

    try:
        from importlib.metadata import entry_points
    except (ImportError, ModuleNotFoundError):  # pragma: no cover
        return

    eps = entry_points()
    plugins = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])

    for ep in plugins:
        try:
            obj = ep.load()
        except Exception as exc:
            log.warning("[Plugins] Failed to load '%s' from group %r: %s", ep.name, group, exc)
            continue
        yield ep.name, obj
