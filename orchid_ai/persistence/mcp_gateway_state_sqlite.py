"""SQLite-backed MCP-gateway-state store (Phase 3).

Implements all three ABCs in
:mod:`orchid_ai.core.mcp_gateway_state` against the tables created by
migration v003.  A single class owns the connection pool; callers
reference one of three narrow interface views
(``ClientStore`` / ``AuthCodeStore`` / ``TokenStore``) via the
read-through accessors.

Configuration::

    MCP_GATEWAY_STATE_STORE_CLASS=orchid_ai.persistence.mcp_gateway_state_sqlite.OrchidSQLiteMCPGatewayStateStore
    MCP_GATEWAY_STATE_STORE_DSN=~/.orchid/chats.db

The DSN is typically shared with the chat storage and MCP-token
stores — operators only maintain one database.

JSON columns are stored as plain ``TEXT`` on SQLite; serialization
happens in this module (``json.dumps`` / ``json.loads``) so the
gateway-side dataclasses never see a string wrapper.  PostgreSQL
uses native ``JSONB`` — see :mod:`mcp_gateway_state_postgres`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import aiosqlite

from ..core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayToken,
    OrchidMCPGatewayTokenStore,
)
from .sqlite import OrchidSQLiteMigrationRunner

logger = logging.getLogger(__name__)


class OrchidSQLiteMCPGatewayStateStore(
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayTokenStore,
):
    """Unified SQLite backend for the three MCP-gateway-state ABCs.

    One connection covers all three concerns because (a) they share a
    migration and (b) the write volume is modest (OAuth flows are
    rare compared to tool calls).  ``aiosqlite.Connection`` is
    thread-safe across coroutines via WAL journalling + per-call
    commits — matches the sibling chat / token / registration stores.

    Atomicity: :meth:`consume` uses a ``BEGIN IMMEDIATE``
    transaction to grab the writer lock before the SELECT, so a
    concurrent caller on a second connection can't observe the row
    between our SELECT and DELETE.  Without this a multi-replica
    gateway could double-spend an auth code.
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None) -> None:
        self._db_path = os.path.expanduser(dsn)
        self._conn: aiosqlite.Connection | None = None
        self._migrator = OrchidSQLiteMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._migrator.run_up(self._conn)
        logger.info("[OrchidMCPGatewayStateStore:sqlite] Initialised — %s", self._db_path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ── Clients ──────────────────────────────────────────────

    async def register(self, record: OrchidMCPGatewayClient) -> None:
        conn = self._require_conn()
        await conn.execute(
            "INSERT OR REPLACE INTO mcp_gateway_clients "
            "(client_id, client_name, redirect_uris, grant_types, response_types, "
            " token_endpoint_auth_method, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.client_id,
                record.client_name,
                json.dumps(record.redirect_uris),
                json.dumps(record.grant_types),
                json.dumps(record.response_types),
                record.token_endpoint_auth_method,
                float(record.created_at),
            ),
        )
        await conn.commit()

    async def get(self, client_id: str) -> OrchidMCPGatewayClient | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM mcp_gateway_clients WHERE client_id = ?",
            (client_id,),
        )
        row = await cursor.fetchone()
        return _row_to_client(row) if row else None

    # ── Auth codes ───────────────────────────────────────────

    async def put(self, record: OrchidMCPGatewayAuthCode) -> None:
        conn = self._require_conn()
        await conn.execute(
            "INSERT INTO mcp_gateway_auth_codes "
            "(code, client_id, redirect_uri, code_challenge, code_challenge_method, "
            " upstream_state, upstream_code_verifier, scopes, client_state, identity, "
            " idp_access_token, idp_refresh_token, idp_expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.code,
                record.client_id,
                record.redirect_uri,
                record.code_challenge,
                record.code_challenge_method,
                record.upstream_state,
                record.upstream_code_verifier,
                json.dumps(record.scopes),
                record.client_state,
                json.dumps(record.identity) if record.identity is not None else None,
                record.idp_access_token,
                record.idp_refresh_token,
                float(record.idp_expires_at),
                float(record.created_at),
            ),
        )
        await conn.commit()

    async def get_by_upstream_state(
        self,
        upstream_state: str,
    ) -> OrchidMCPGatewayAuthCode | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM mcp_gateway_auth_codes WHERE upstream_state = ?",
            (upstream_state,),
        )
        row = await cursor.fetchone()
        return _row_to_auth_code(row) if row else None

    async def update(
        self,
        code: str,
        *,
        identity: dict[str, Any] | None = None,
        idp_access_token: str | None = None,
        idp_refresh_token: str | None = None,
        idp_expires_at: float | None = None,
    ) -> None:
        conn = self._require_conn()
        # Build a partial UPDATE — skip any column the caller omitted
        # so we don't accidentally wipe a field set by a previous
        # partial update.
        sets: list[str] = []
        params: list[Any] = []
        if identity is not None:
            sets.append("identity = ?")
            params.append(json.dumps(identity))
        if idp_access_token is not None:
            sets.append("idp_access_token = ?")
            params.append(idp_access_token)
        if idp_refresh_token is not None:
            sets.append("idp_refresh_token = ?")
            params.append(idp_refresh_token)
        if idp_expires_at is not None:
            sets.append("idp_expires_at = ?")
            params.append(float(idp_expires_at))
        if not sets:
            return
        params.append(code)
        await conn.execute(
            f"UPDATE mcp_gateway_auth_codes SET {', '.join(sets)} WHERE code = ?",
            params,
        )
        await conn.commit()

    async def consume(self, code: str) -> OrchidMCPGatewayAuthCode | None:
        conn = self._require_conn()
        # ``BEGIN IMMEDIATE`` takes the writer lock up-front; a
        # concurrent ``consume`` on a second connection blocks here
        # rather than racing our SELECT.  One-shot semantics intact.
        await conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await conn.execute(
                "SELECT * FROM mcp_gateway_auth_codes WHERE code = ?",
                (code,),
            )
            row = await cursor.fetchone()
            if row is None:
                await conn.execute("COMMIT")
                return None
            await conn.execute(
                "DELETE FROM mcp_gateway_auth_codes WHERE code = ?",
                (code,),
            )
            await conn.execute("COMMIT")
        except Exception:
            await conn.execute("ROLLBACK")
            raise
        return _row_to_auth_code(row)

    # ── Tokens ──────────────────────────────────────────────

    async def issue(self, record: OrchidMCPGatewayToken) -> None:
        conn = self._require_conn()
        await conn.execute(
            "INSERT INTO mcp_gateway_tokens "
            "(access_token, refresh_token, client_id, subject, identity, scopes, expires_at, "
            " idp_access_token, idp_refresh_token, idp_expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.access_token,
                record.refresh_token,
                record.client_id,
                record.subject,
                json.dumps(record.identity),
                json.dumps(record.scopes),
                float(record.expires_at),
                record.idp_access_token,
                record.idp_refresh_token,
                float(record.idp_expires_at),
            ),
        )
        await conn.commit()

    async def get_by_access_token(
        self,
        access_token: str,
    ) -> OrchidMCPGatewayToken | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM mcp_gateway_tokens WHERE access_token = ?",
            (access_token,),
        )
        row = await cursor.fetchone()
        return _not_expired_token(_row_to_token(row) if row else None)

    async def get_by_refresh_token(
        self,
        refresh_token: str,
    ) -> OrchidMCPGatewayToken | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM mcp_gateway_tokens WHERE refresh_token = ?",
            (refresh_token,),
        )
        row = await cursor.fetchone()
        return _not_expired_token(_row_to_token(row) if row else None)

    async def revoke(self, access_token: str) -> bool:
        conn = self._require_conn()
        cursor = await conn.execute(
            "DELETE FROM mcp_gateway_tokens WHERE access_token = ?",
            (access_token,),
        )
        await conn.commit()
        return cursor.rowcount > 0

    # ── Internals ───────────────────────────────────────────

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("init_db() must be called first")
        return self._conn


# ── Row mappers ─────────────────────────────────────────────


def _row_to_client(row: aiosqlite.Row) -> OrchidMCPGatewayClient:
    return OrchidMCPGatewayClient(
        client_id=row["client_id"],
        client_name=row["client_name"],
        redirect_uris=json.loads(row["redirect_uris"]),
        grant_types=json.loads(row["grant_types"]),
        response_types=json.loads(row["response_types"]),
        token_endpoint_auth_method=row["token_endpoint_auth_method"],
        created_at=float(row["created_at"]),
    )


def _row_to_auth_code(row: aiosqlite.Row) -> OrchidMCPGatewayAuthCode:
    identity_raw = row["identity"]
    return OrchidMCPGatewayAuthCode(
        code=row["code"],
        client_id=row["client_id"],
        redirect_uri=row["redirect_uri"],
        code_challenge=row["code_challenge"],
        code_challenge_method=row["code_challenge_method"],
        upstream_state=row["upstream_state"],
        upstream_code_verifier=row["upstream_code_verifier"],
        scopes=json.loads(row["scopes"]),
        client_state=row["client_state"],
        identity=json.loads(identity_raw) if identity_raw else None,
        idp_access_token=row["idp_access_token"],
        idp_refresh_token=row["idp_refresh_token"],
        idp_expires_at=float(row["idp_expires_at"]),
        created_at=float(row["created_at"]),
    )


def _row_to_token(row: aiosqlite.Row) -> OrchidMCPGatewayToken:
    return OrchidMCPGatewayToken(
        access_token=row["access_token"],
        refresh_token=row["refresh_token"],
        client_id=row["client_id"],
        subject=row["subject"],
        identity=json.loads(row["identity"]),
        scopes=json.loads(row["scopes"]),
        expires_at=float(row["expires_at"]),
        idp_access_token=row["idp_access_token"] or "",
        idp_refresh_token=row["idp_refresh_token"] or "",
        idp_expires_at=float(row["idp_expires_at"] or 0.0),
    )


def _not_expired_token(record: OrchidMCPGatewayToken | None) -> OrchidMCPGatewayToken | None:
    if record is None:
        return None
    if record.expires_at > 0 and time.time() >= record.expires_at:
        return None
    return record
