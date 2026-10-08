"""
Factory for MCP token storage backends.

Resolves a dotted class path to a concrete ``OrchidMCPTokenStore`` implementation
and instantiates it.  The library ships a dependency-free in-memory backend
(default); durable backends are available via plugins
(``orchid-storage-sqlite``, ``orchid-storage-postgres``) or consumer projects
via dotted import paths.

The ``"memory"`` sentinel (also the empty string) selects the built-in
in-memory backend.  A configured dotted class path always wins over the
default and fails strictly when it cannot be imported.

Usage:
    store = build_mcp_token_store("memory", dsn="")
    await store.init_db()
"""

from __future__ import annotations

import logging

from ..core.mcp import OrchidMCPTokenStore
from ..utils import import_class, is_memory_storage_sentinel
from .in_memory import OrchidInMemoryMCPTokenStore
from .plugin_hints import with_plugin_hint

logger = logging.getLogger(__name__)


def build_mcp_token_store(
    class_path: str,
    dsn: str,
    *,
    extra_migrations_package: str | None = None,
) -> OrchidMCPTokenStore:
    """
    Dynamically import and instantiate an OrchidMCPTokenStore backend.

    Parameters
    ----------
    class_path : str
        Dotted path to an ``OrchidMCPTokenStore`` subclass, or the
        ``"memory"`` sentinel (also the empty string) for the built-in
        in-memory backend.
        Example: ``"orchid_storage_postgres.OrchidPostgresMCPTokenStore"``
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
        Passed as ``dsn=`` keyword to the constructor.  Ignored by the
        in-memory backend.
    extra_migrations_package : str | None
        Optional dotted import path of an integrator-supplied migrations
        package.  When provided, those migrations run after the
        framework's (see
        :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`).
        Ignored by the in-memory backend.

    Returns
    -------
    OrchidMCPTokenStore
        An uninitialised instance — caller must ``await .init_db()``.
    """
    if is_memory_storage_sentinel(class_path):
        logger.info("[OrchidMCPTokenStore] Using built-in in-memory backend")
        return OrchidInMemoryMCPTokenStore(dsn=dsn, extra_migrations_package=extra_migrations_package)

    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            with_plugin_hint(
                f"Cannot resolve MCP token store class '{class_path}'. "
                f"Ensure it is a valid dotted import path to an OrchidMCPTokenStore subclass. "
                f"Error: {exc}",
                class_path,
            )
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, OrchidMCPTokenStore)):
        raise TypeError(f"'{class_path}' resolves to {cls!r}, which is not an OrchidMCPTokenStore subclass.")

    logger.info("[OrchidMCPTokenStore] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
