"""Factory for :class:`OrchidMCPClientRegistrationStore` backends.

Mirrors :func:`build_mcp_token_store`: resolves a dotted class path and
constructs the backend with a ``dsn`` + optional integrator-migration
hook.  The library ships a dependency-free in-memory backend (default);
durable backends are available via plugins or consumer projects.  The
``"memory"`` sentinel (also the empty string) selects the in-memory
backend; configured dotted paths resolve strictly.
"""

from __future__ import annotations

import logging

from ..core.mcp import OrchidMCPClientRegistrationStore
from ..utils import import_class, is_memory_storage_sentinel
from .in_memory import OrchidInMemoryMCPClientRegistrationStore
from .plugin_hints import with_plugin_hint

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
        Dotted path to an :class:`OrchidMCPClientRegistrationStore`
        subclass, or the ``"memory"`` sentinel (also the empty string)
        for the built-in in-memory backend, e.g.
        ``"orchid_storage_postgres.OrchidPostgresMCPClientRegistrationStore"``.
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
        Ignored by the in-memory backend.
    extra_migrations_package : str | None
        Optional dotted path to an integrator migrations package —
        appended after the framework's per
        :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`.
        Ignored by the in-memory backend.
    """
    if is_memory_storage_sentinel(class_path):
        logger.info("[OrchidMCPClientRegistrationStore] Using built-in in-memory backend")
        return OrchidInMemoryMCPClientRegistrationStore(dsn=dsn, extra_migrations_package=extra_migrations_package)

    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            with_plugin_hint(
                f"Cannot resolve MCP client-registration store class '{class_path}'. "
                f"Ensure it is a valid dotted import path to an "
                f"OrchidMCPClientRegistrationStore subclass.  Error: {exc}",
                class_path,
            )
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, OrchidMCPClientRegistrationStore)):
        raise TypeError(
            f"'{class_path}' resolves to {cls!r}, which is not an OrchidMCPClientRegistrationStore subclass."
        )

    logger.info("[OrchidMCPClientRegistrationStore] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
