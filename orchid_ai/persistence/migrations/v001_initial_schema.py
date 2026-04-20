"""
Migration v001 — Initial persistence schema (chat + MCP OAuth tokens).

Creates:
  - chat_sessions table
  - chat_messages table (with FK cascade to sessions)
  - mcp_oauth_tokens table (per-server OAuth tokens, same DB)
  - Supporting indices for user listing, message ordering, and token lookup

Dialect-aware: uses TIMESTAMPTZ/JSONB/DOUBLE PRECISION on PostgreSQL,
TEXT/REAL on SQLite.
"""

VERSION = "001"
DESCRIPTION = "Initial schema: chat sessions, messages, and MCP OAuth tokens"

# ── PostgreSQL DDL ──────────────────────────────────────────

_PG_UP = [
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        is_shared BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_user
        ON chat_sessions (tenant_id, user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        agents_used JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_chat
        ON chat_messages (chat_id, created_at ASC)
    """,
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
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        is_shared INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_user
        ON chat_sessions (tenant_id, user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        agents_used TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_chat
        ON chat_messages (chat_id, created_at ASC)
    """,
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
    "DROP TABLE IF EXISTS chat_messages",
    "DROP TABLE IF EXISTS chat_sessions",
]


async def up(conn, *, dialect: str = "postgres") -> None:
    stmts = _SQLITE_UP if dialect == "sqlite" else _PG_UP
    for sql in stmts:
        await conn.execute(sql)


async def down(conn, *, dialect: str = "postgres") -> None:
    for sql in _DOWN:
        await conn.execute(sql)
