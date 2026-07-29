"""
Entry-point plugin discovery — shared by orchid-api, orchid-cli, and
anything else that wants to load optional hooks from installed packages.

The function :func:`iter_entry_point_plugins` walks a named entry-point
group and yields ``(name, loaded_object)`` pairs, isolating the caller
from the minor API differences between Python 3.9–3.11 and 3.12+.

:func:`lazy_init_plugins` loads all framework-managed plugin groups
(vector backends, doc stores, graph stores, checkpointers, visibility
fragments) on first call — NOT at module import time.  Call it once
at the application entry point (``Orchid.__init__`` calls it
automatically).

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
from collections.abc import Iterator
from typing import Any

__all__ = ["iter_entry_point_plugins", "lazy_init_plugins"]

_lazy_init_done: bool = False


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


def lazy_init_plugins() -> None:
    """Load all framework-managed plugins once.

    Called from :class:`orchid_ai.Orchid.__init__` automatically.
    Safe to call multiple times — only the first call does work.
    """
    global _lazy_init_done
    if _lazy_init_done:
        return
    _lazy_init_done = True

    _log = logging.getLogger(__name__)
    _log.debug("[Plugins] Loading framework-managed entry points")

    # RAG backends (vector, doc_store, graph_store)
    from .rag.factory import _load_entry_point_backends

    _load_entry_point_backends()

    # Checkpointers
    from .checkpointing.factory import _load_entry_point_checkpointers

    _load_entry_point_checkpointers()

    # Visibility fragments
    from .events.visibility import _load_entry_point_visibility_fragments

    _load_entry_point_visibility_fragments()
