"""Factory for :class:`OrchidMCPClientRegistrationStore` backends.

Mirrors :func:`build_mcp_token_store`: resolves a dotted class path and
constructs the backend with a ``dsn`` + optional integrator-migration
hook.  The library ships SQLite (default) and PostgreSQL backends;
consumers override by pointing the class path at any subclass.
"""

from __future__ import annotations

import logging

from ..core.mcp import OrchidMCPClientRegistrationStore
from ..utils import import_class

logger = logging.getLogger(__name__)


def build_mcp_client_registration_store(
    class_path: str,
    dsn: str,
    *,
    extra_migrations_package: str | None = None,
) -> OrchidMCPClientRegistrationStore:
    """Dynamically import and instantiate a registration-store backend.

    Parameters
    ----------
    class_path : str
        Fully-qualified dotted path to an
        :class:`OrchidMCPClientRegistrationStore` subclass, e.g.
        ``"orchid_ai.persistence.mcp_client_registration_sqlite.OrchidSQLiteMCPClientRegistrationStore"``.
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
    extra_migrations_package : str | None
        Optional dotted path to an integrator migrations package —
        appended after the framework's per
        :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`.
    """
    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve MCP client-registration store class '{class_path}'. "
            f"Ensure it is a valid dotted import path to an "
            f"OrchidMCPClientRegistrationStore subclass.  Error: {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, OrchidMCPClientRegistrationStore)):
        raise TypeError(
            f"'{class_path}' resolves to {cls!r}, which is not an OrchidMCPClientRegistrationStore subclass."
        )

    logger.info("[OrchidMCPClientRegistrationStore] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
