"""
Factory for MCP token storage backends.

Resolves a dotted class path to a concrete ``MCPTokenStore`` implementation
and instantiates it.  The library ships built-in SQLite (default) and
PostgreSQL backends; consumers can provide alternative backends via
dotted import paths.

Usage:
    store = build_mcp_token_store(
        class_path="orchid_ai.persistence.mcp_token_sqlite.SQLiteMCPTokenStore",
        dsn="~/.orchid/mcp_tokens.db",
    )
    await store.init_db()
"""

from __future__ import annotations

import logging

from ..core.mcp import MCPTokenStore
from ..utils import import_class

logger = logging.getLogger(__name__)


def build_mcp_token_store(class_path: str, dsn: str) -> MCPTokenStore:
    """
    Dynamically import and instantiate an MCPTokenStore backend.

    Parameters
    ----------
    class_path : str
        Fully-qualified dotted path to an ``MCPTokenStore`` subclass.
        Example: ``"orchid_ai.persistence.mcp_token_sqlite.SQLiteMCPTokenStore"``
    dsn : str
        Connection string (PostgreSQL DSN) or file path (SQLite).
        Passed as ``dsn=`` keyword to the constructor.

    Returns
    -------
    MCPTokenStore
        An uninitialised instance — caller must ``await .init_db()``.
    """
    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve MCP token store class '{class_path}'. "
            f"Ensure it is a valid dotted import path to an MCPTokenStore subclass. "
            f"Error: {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, MCPTokenStore)):
        raise TypeError(f"'{class_path}' resolves to {cls!r}, which is not an MCPTokenStore subclass.")

    logger.info("[MCPTokenStore] Using %s", class_path)
    return cls(dsn=dsn)
