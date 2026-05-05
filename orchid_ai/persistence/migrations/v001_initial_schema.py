"""
Migration v001 — Initial persistence schema (unified).

Creates every framework-owned table in a single pass:

**Chat persistence**
  - ``chat_sessions`` — one row per user chat thread.
  - ``chat_messages`` — conversation turns, FK-cascade to sessions.

**MCP outbound (orchid-api as OAuth client to external MCP servers)**
  - ``mcp_oauth_tokens`` — per-user access + refresh tokens.
  - ``mcp_client_registrations`` — per-server discovered auth
    endpoints + DCR (RFC 7591) credentials.

**MCP inbound gateway state (external MCP clients authenticating to
the gateway via OAuth 2.0 + DCR)**
  - ``mcp_gateway_clients`` — registered inbound DCR clients.
  - ``mcp_gateway_auth_codes`` — in-flight authorization codes with
    upstream-IdP correlation state.
  - ``mcp_gateway_tokens`` — issued gateway access + refresh tokens
    with the resolved identity payload.

Dialect-aware: uses ``TIMESTAMPTZ`` + ``JSONB`` + ``DOUBLE PRECISION``
on PostgreSQL; ``TEXT`` + ``REAL`` on SQLite.  The JSON columns on
SQLite store serialized strings — callers (the respective store
backends) own the ``json.dumps`` / ``json.loads`` boundary.
"""

VERSION = "001"
DESCRIPTION = "Unified initial schema (chat, MCP outbound, MCP inbound gateway)"


# ── PostgreSQL DDL ──────────────────────────────────────────

_PG_UP = [
    # ── Chat persistence ──────────────────────────────
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
    # ── MCP outbound: per-user OAuth tokens ──────────
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
    # ── MCP outbound: per-server DCR registrations ──
    """
    CREATE TABLE IF NOT EXISTS mcp_client_registrations (
        server_name                             TEXT PRIMARY KEY,
        authorization_endpoint                  TEXT NOT NULL,
        token_endpoint                          TEXT NOT NULL,
        registration_endpoint                   TEXT NOT NULL DEFAULT '',
        issuer                                  TEXT NOT NULL DEFAULT '',
        scopes_supported                        TEXT NOT NULL DEFAULT '',
        token_endpoint_auth_methods_supported   TEXT NOT NULL DEFAULT 'client_secret_post',
        client_id                               TEXT NOT NULL DEFAULT '',
        client_secret                           TEXT NOT NULL DEFAULT '',
        client_id_issued_at                     DOUBLE PRECISION NOT NULL DEFAULT 0,
        client_secret_expires_at                DOUBLE PRECISION NOT NULL DEFAULT 0,
        created_at                              TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at                              TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # ── MCP inbound gateway state ─────────────────────
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_clients (
        client_id                      TEXT PRIMARY KEY,
        client_name                    TEXT NOT NULL DEFAULT '',
        redirect_uris                  JSONB NOT NULL,
        grant_types                    JSONB NOT NULL,
        response_types                 JSONB NOT NULL,
        token_endpoint_auth_method     TEXT NOT NULL DEFAULT 'none',
        created_at                     DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_auth_codes (
        code                           TEXT PRIMARY KEY,
        client_id                      TEXT NOT NULL,
        redirect_uri                   TEXT NOT NULL,
        code_challenge                 TEXT NOT NULL,
        code_challenge_method          TEXT NOT NULL,
        upstream_state                 TEXT NOT NULL UNIQUE,
        upstream_code_verifier         TEXT NOT NULL,
        scopes                         JSONB NOT NULL,
        client_state                   TEXT NOT NULL DEFAULT '',
        identity                       JSONB,
        idp_access_token               TEXT NOT NULL DEFAULT '',
        idp_refresh_token              TEXT NOT NULL DEFAULT '',
        idp_expires_at                 DOUBLE PRECISION NOT NULL DEFAULT 0,
        created_at                     DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_gateway_auth_codes_created_at
        ON mcp_gateway_auth_codes (created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_tokens (
        access_token                   TEXT PRIMARY KEY,
        refresh_token                  TEXT NOT NULL UNIQUE,
        client_id                      TEXT NOT NULL,
        subject                        TEXT NOT NULL,
        identity                       JSONB NOT NULL,
        scopes                         JSONB NOT NULL,
        expires_at                     DOUBLE PRECISION NOT NULL,
        -- Upstream IdP tokens carried alongside so the gateway's
        -- refresh flow can rotate them without a fresh browser-based
        -- re-authentication.  Empty string / 0.0 defaults cover
        -- legacy records written before this schema went live.
        idp_access_token               TEXT NOT NULL DEFAULT '',
        idp_refresh_token              TEXT NOT NULL DEFAULT '',
        idp_expires_at                 DOUBLE PRECISION NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_gateway_tokens_expires_at
        ON mcp_gateway_tokens (expires_at)
    """,
]


# ── SQLite DDL ──────────────────────────────────────────────

_SQLITE_UP = [
    # ── Chat persistence ──────────────────────────────
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
    # ── MCP outbound: per-user OAuth tokens ──────────
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
    # ── MCP outbound: per-server DCR registrations ──
    """
    CREATE TABLE IF NOT EXISTS mcp_client_registrations (
        server_name                             TEXT PRIMARY KEY,
        authorization_endpoint                  TEXT NOT NULL,
        token_endpoint                          TEXT NOT NULL,
        registration_endpoint                   TEXT NOT NULL DEFAULT '',
        issuer                                  TEXT NOT NULL DEFAULT '',
        scopes_supported                        TEXT NOT NULL DEFAULT '',
        token_endpoint_auth_methods_supported   TEXT NOT NULL DEFAULT 'client_secret_post',
        client_id                               TEXT NOT NULL DEFAULT '',
        client_secret                           TEXT NOT NULL DEFAULT '',
        client_id_issued_at                     REAL NOT NULL DEFAULT 0,
        client_secret_expires_at                REAL NOT NULL DEFAULT 0,
        created_at                              TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at                              TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # ── MCP inbound gateway state ─────────────────────
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_clients (
        client_id                      TEXT PRIMARY KEY,
        client_name                    TEXT NOT NULL DEFAULT '',
        redirect_uris                  TEXT NOT NULL,
        grant_types                    TEXT NOT NULL,
        response_types                 TEXT NOT NULL,
        token_endpoint_auth_method     TEXT NOT NULL DEFAULT 'none',
        created_at                     REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_auth_codes (
        code                           TEXT PRIMARY KEY,
        client_id                      TEXT NOT NULL,
        redirect_uri                   TEXT NOT NULL,
        code_challenge                 TEXT NOT NULL,
        code_challenge_method          TEXT NOT NULL,
        upstream_state                 TEXT NOT NULL UNIQUE,
        upstream_code_verifier         TEXT NOT NULL,
        scopes                         TEXT NOT NULL,
        client_state                   TEXT NOT NULL DEFAULT '',
        identity                       TEXT,
        idp_access_token               TEXT NOT NULL DEFAULT '',
        idp_refresh_token              TEXT NOT NULL DEFAULT '',
        idp_expires_at                 REAL NOT NULL DEFAULT 0,
        created_at                     REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_gateway_auth_codes_created_at
        ON mcp_gateway_auth_codes (created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_gateway_tokens (
        access_token                   TEXT PRIMARY KEY,
        refresh_token                  TEXT NOT NULL UNIQUE,
        client_id                      TEXT NOT NULL,
        subject                        TEXT NOT NULL,
        identity                       TEXT NOT NULL,
        scopes                         TEXT NOT NULL,
        expires_at                     REAL NOT NULL,
        -- Upstream-token columns; see Postgres block above.
        idp_access_token               TEXT NOT NULL DEFAULT '',
        idp_refresh_token              TEXT NOT NULL DEFAULT '',
        idp_expires_at                 REAL NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mcp_gateway_tokens_expires_at
        ON mcp_gateway_tokens (expires_at)
    """,
]


# Reverse order so FK-dependent tables drop before their referents.
_DOWN = [
    "DROP TABLE IF EXISTS mcp_gateway_tokens",
    "DROP TABLE IF EXISTS mcp_gateway_auth_codes",
    "DROP TABLE IF EXISTS mcp_gateway_clients",
    "DROP TABLE IF EXISTS mcp_client_registrations",
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
