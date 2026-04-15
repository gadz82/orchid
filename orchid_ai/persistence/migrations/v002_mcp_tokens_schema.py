"""
Migration v002 — MCP OAuth tokens schema.

Creates:
  - mcp_oauth_tokens table (PK on server_name, tenant_id, user_id)
  - Index for listing all tokens by tenant + user

Dialect-aware: uses TIMESTAMPTZ/DOUBLE PRECISION on PostgreSQL,
TEXT/REAL on SQLite.
"""

VERSION = "002"
DESCRIPTION = "MCP OAuth tokens schema"

# ── PostgreSQL DDL ──────────────────────────────────────────

_PG_UP = [
    """
    CREATE TABLE IF NOT EXISTS mcp_oauth_tokens (
        server_name  TEXT NOT NULL,
        tenant_id    TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT NOT NULL DEFAULT '',
        expires_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
        scopes       TEXT NOT NULL DEFAULT '',
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (server_name, tenant_id, user_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_tokens_user
        ON mcp_oauth_tokens (tenant_id, user_id)
    """,
]

# ── SQLite DDL ──────────────────────────────────────────────

_SQLITE_UP = [
    """
    CREATE TABLE IF NOT EXISTS mcp_oauth_tokens (
        server_name  TEXT NOT NULL,
        tenant_id    TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT NOT NULL DEFAULT '',
        expires_at   REAL NOT NULL DEFAULT 0,
        scopes       TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (server_name, tenant_id, user_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_tokens_user
        ON mcp_oauth_tokens (tenant_id, user_id)
    """,
]

_DOWN = [
    "DROP TABLE IF EXISTS mcp_oauth_tokens",
]


async def up(conn, *, dialect: str = "postgres") -> None:
    stmts = _SQLITE_UP if dialect == "sqlite" else _PG_UP
    for sql in stmts:
        await conn.execute(sql)


async def down(conn, *, dialect: str = "postgres") -> None:
    for sql in _DOWN:
        await conn.execute(sql)
