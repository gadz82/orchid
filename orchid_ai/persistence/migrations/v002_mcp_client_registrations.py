"""
Migration v002 — per-server MCP client registrations (RFC 7591 DCR).

Creates the ``mcp_client_registrations`` table used by
:class:`orchid_ai.core.mcp.OrchidMCPClientRegistrationStore`.  One row
per MCP server: the authorization-server endpoints discovered via
:rfc:`8414` + the dynamically-registered client credentials obtained
from :rfc:`7591`.  This is separate from ``mcp_oauth_tokens`` (which
stores the per-user access + refresh tokens) because the client
registration is a property of the server, not of the user.

Dialect-aware: uses TIMESTAMPTZ/DOUBLE PRECISION on PostgreSQL, TEXT/REAL
on SQLite — matches the conventions in v001.
"""

VERSION = "002"
DESCRIPTION = "Per-server MCP client registrations (RFC 7591 DCR)"


# ── PostgreSQL DDL ──────────────────────────────────────────

_PG_UP = [
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
]


# ── SQLite DDL ──────────────────────────────────────────────

_SQLITE_UP = [
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
]


_DOWN = [
    "DROP TABLE IF EXISTS mcp_client_registrations",
]


async def up(conn, *, dialect: str = "postgres") -> None:
    stmts = _SQLITE_UP if dialect == "sqlite" else _PG_UP
    for sql in stmts:
        await conn.execute(sql)


async def down(conn, *, dialect: str = "postgres") -> None:
    for sql in _DOWN:
        await conn.execute(sql)
