"""
Factory for chat storage backends.

Resolves a dotted class path to a concrete ``ChatStorage`` implementation
and instantiates it.  The library ships built-in SQLite (default) and
PostgreSQL backends; consumers can provide alternative backends via
dotted import paths.

Usage:
    storage = build_chat_storage(
        class_path="orchid.persistence.sqlite.SQLiteChatStorage",
        dsn="~/.orchid/chats.db",
    )
    await storage.init_db()
"""

from __future__ import annotations

import logging

from ..utils import import_class
from .base import ChatStorage

logger = logging.getLogger(__name__)

# Backward-compat alias for any code importing _import_class from here
_import_class = import_class


def build_chat_storage(class_path: str, dsn: str) -> ChatStorage:
    """
    Dynamically import and instantiate a ChatStorage backend.

    Parameters
    ----------
    class_path : str
        Fully-qualified dotted path to a ``ChatStorage`` subclass.
        Example: ``"orchid.persistence.postgres.PostgresChatStorage"``
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
        Passed as ``dsn=`` keyword to the constructor.

    Returns
    -------
    ChatStorage
        An uninitialised instance — caller must ``await .init_db()``.
    """
    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve chat storage class '{class_path}'. "
            f"Ensure it is a valid dotted import path to a ChatStorage subclass. "
            f"Error: {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, ChatStorage)):
        raise TypeError(
            f"'{class_path}' resolves to {cls!r}, which is not a ChatStorage subclass."
        )

    logger.info("[ChatStorage] Using %s", class_path)
    return cls(dsn=dsn)
