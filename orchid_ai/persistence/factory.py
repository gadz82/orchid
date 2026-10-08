"""
Factory for chat storage backends.

Resolves a dotted class path to a concrete ``OrchidChatStorage`` implementation
and instantiates it.  The library ships a dependency-free in-memory backend
(default); durable backends are available via plugins
(``orchid-storage-sqlite``, ``orchid-storage-postgres``) or consumer projects
via dotted import paths.

The ``"memory"`` sentinel (also the empty string) selects the built-in
in-memory backend.  A configured dotted class path always wins over the
default and fails strictly when it cannot be imported — there is no silent
fallback to in-memory.

Usage:
    storage = build_chat_storage("memory", dsn="")
    await storage.init_db()
"""

from __future__ import annotations

import logging

from ..utils import import_class, is_memory_storage_sentinel
from .base import OrchidChatStorage
from .in_memory import OrchidInMemoryChatStorage
from .plugin_hints import with_plugin_hint

logger = logging.getLogger(__name__)


def _augment_import_error(class_path: str, exc: Exception) -> ImportError:
    msg = (
        f"Cannot resolve chat storage class '{class_path}'. "
        f"Ensure it is a valid dotted import path to a OrchidChatStorage subclass. "
        f"Error: {exc}"
    )
    return ImportError(with_plugin_hint(msg, class_path))


def build_chat_storage(
    class_path: str,
    dsn: str,
    *,
    extra_migrations_package: str | None = None,
) -> OrchidChatStorage:
    """
    Dynamically import and instantiate a OrchidChatStorage backend.

    Parameters
    ----------
    class_path : str
        Dotted path to a ``OrchidChatStorage`` subclass, or the
        ``"memory"`` sentinel (also the empty string) for the built-in
        in-memory backend.
        Example: ``"orchid_storage_postgres.OrchidPostgresChatStorage"``
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
    OrchidChatStorage
        An uninitialised instance — caller must ``await .init_db()``.
    """
    if is_memory_storage_sentinel(class_path):
        logger.info("[OrchidChatStorage] Using built-in in-memory backend")
        return OrchidInMemoryChatStorage(dsn=dsn, extra_migrations_package=extra_migrations_package)

    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise _augment_import_error(class_path, exc) from exc

    if not (isinstance(cls, type) and issubclass(cls, OrchidChatStorage)):
        raise TypeError(f"'{class_path}' resolves to {cls!r}, which is not a OrchidChatStorage subclass.")

    logger.info("[OrchidChatStorage] Using %s", class_path)
    return cls(dsn=dsn, extra_migrations_package=extra_migrations_package)
