"""Migration v002 - ensure framework-owned tables exist.

Some deployments already have a ``_migrations`` row for version ``001``
from an older or custom chat-storage migration.  The unified framework
``v001`` now owns chat, MCP, and events tables, but a pre-existing
``001`` record makes the migration runner skip it.  Replaying the
idempotent v001 DDL under a new framework version repairs those
databases and keeps fresh databases unchanged.
"""

from __future__ import annotations

from .v001_initial_schema import _PG_UP, _SQLITE_UP

VERSION = "002"
DESCRIPTION = "Ensure framework-owned schema exists after legacy v001 records"


async def up(conn, *, dialect: str = "postgres") -> None:
    stmts = _SQLITE_UP if dialect == "sqlite" else _PG_UP
    for sql in stmts:
        await conn.execute(sql)


async def down(conn, *, dialect: str = "postgres") -> None:
    """No-op: v001 remains the owner of the idempotently replayed tables."""
    return None
