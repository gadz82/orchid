"""
Factory for MCP token storage backends.

Resolves a dotted class path to a concrete ``OrchidMCPTokenStore`` implementation
and instantiates it.  The library ships built-in SQLite (default) and
PostgreSQL backends; consumers can provide alternative backends via
dotted import paths.

Usage:
    store = build_mcp_token_store(
        class_path="orchid_ai.persistence.mcp_token_sqlite.OrchidSQLiteMCPTokenStore",
        dsn="~/.orchid/mcp_tokens.db",
    )
    await store.init_db()
"""

from __future__ import annotations

import logging

from ..core.mcp import OrchidMCPTokenStore
from ..utils import import_class

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
        Fully-qualified dotted path to an ``OrchidMCPTokenStore`` subclass.
        Example: ``"orchid_ai.persistence.mcp_token_sqlite.OrchidSQLiteMCPTokenStore"``
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
        Passed as ``dsn=`` keyword to the constructor.
    extra_migrations_package : str | None
        Optional dotted import path of an integrator-supplied migrations
        package.  When provided, those migrations run after the
        framework's (see
        :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`).

    Returns
    -------
    OrchidMCPTokenStore
        An uninitialised instance — caller must ``await .init_db()``.
    """
    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve MCP token store class '{class_path}'. "
            f"Ensure it is a valid dotted import path to an OrchidMCPTokenStore subclass. "
            f"Error: {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, OrchidMCPTokenStore)):
        raise TypeError(f"'{class_path}' resolves to {cls!r}, which is not an OrchidMCPTokenStore subclass.")

    logger.info("[OrchidMCPTokenStore] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
