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

**Pollen + Bloom (event-driven activation layer)**
  - ``signals`` — append-only normalised events.
  - ``signal_queue`` — transient queue between dispatcher and
    processor, with leases.
  - ``signal_queue_dead_letter`` — terminal poisoned messages.
  - ``triggers`` — versioned trigger configs.
  - ``schedules`` — cron / interval entries owned by the
    scheduler producer.
  - ``job_runs`` — one row per attempt at running a JobSpec; the
    ``UNIQUE (trigger_id, signal_id, attempt_number)`` constraint
    is what makes Bloom idempotent under queue redelivery.
  - ``signal_sources`` — webhook source registry consumed by the
    HTTP ingestion producer.

Dialect-aware: uses ``TIMESTAMPTZ`` + ``JSONB`` + ``DOUBLE PRECISION``
on PostgreSQL; ``TEXT`` + ``REAL`` on SQLite.  The JSON columns on
SQLite store serialized strings — callers (the respective store
backends) own the ``json.dumps`` / ``json.loads`` boundary.
"""

from __future__ import annotations

from ._schema_ddl import PG_UP, SQLITE_UP

VERSION = "001"
DESCRIPTION = "Unified initial schema (chat, MCP outbound, MCP inbound gateway, events)"

# Aliases for backward compat with any external code that imported these.
_PG_UP = PG_UP
_SQLITE_UP = SQLITE_UP


# Reverse order so FK-dependent tables drop before their referents.
_DOWN = [
    "DROP TABLE IF EXISTS signal_sources",
    "DROP TABLE IF EXISTS job_runs",
    "DROP TABLE IF EXISTS schedules",
    "DROP TABLE IF EXISTS triggers",
    "DROP TABLE IF EXISTS signal_queue_dead_letter",
    "DROP TABLE IF EXISTS signal_queue",
    "DROP TABLE IF EXISTS signals",
    "DROP TABLE IF EXISTS mcp_gateway_tokens",
    "DROP TABLE IF EXISTS mcp_gateway_auth_codes",
    "DROP TABLE IF EXISTS mcp_gateway_clients",
    "DROP TABLE IF EXISTS mcp_client_registrations",
    "DROP TABLE IF EXISTS mcp_oauth_tokens",
    "DROP TABLE IF EXISTS chat_messages",
    "DROP TABLE IF EXISTS chat_sessions",
]


async def up(conn, *, dialect: str = "postgres") -> None:
    stmts = SQLITE_UP if dialect == "sqlite" else PG_UP
    for sql in stmts:
        await conn.execute(sql)


async def down(conn, *, dialect: str = "postgres") -> None:
    for sql in _DOWN:
        await conn.execute(sql)
