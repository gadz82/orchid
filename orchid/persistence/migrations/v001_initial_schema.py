"""
Migration v001 — Initial chat persistence schema.

Creates:
  - chat_sessions table
  - chat_messages table (with FK cascade to sessions)
  - Indices for user listing and message ordering

Dialect-aware: uses TIMESTAMPTZ/JSONB on PostgreSQL, TEXT on SQLite.
"""

VERSION = "001"
DESCRIPTION = "Initial chat sessions and messages schema"

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
]

_DOWN = [
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
