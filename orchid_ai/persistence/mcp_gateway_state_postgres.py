"""PostgreSQL-backed MCP-gateway-state store (Phase 3).

Implements all three ABCs in :mod:`orchid_ai.core.mcp_gateway_state`
against the tables created by migration v003, using asyncpg.
Uses native ``JSONB`` for the serialized identity / array columns so
operators can ``jsonb_path_query`` into live state during ops.

Atomic :meth:`consume` uses ``DELETE … RETURNING`` — the standard
Postgres idiom for one-shot semantics.  Safe under multi-replica
contention: the row returns to exactly one caller.

Optional dependency — install via ``pip install orchid-ai[postgres]``.

Configuration::

    MCP_GATEWAY_STATE_STORE_CLASS=orchid_ai.persistence.mcp_gateway_state_postgres.OrchidPostgresMCPGatewayStateStore
    MCP_GATEWAY_STATE_STORE_DSN=postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayToken,
    OrchidMCPGatewayTokenStore,
)
from .postgres import OrchidPostgresMigrationRunner

logger = logging.getLogger(__name__)


class OrchidPostgresMCPGatewayStateStore(
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayTokenStore,
):
    """Unified PostgreSQL backend for the three MCP-gateway-state ABCs.

    Pool: ``min_size=2, max_size=10`` to match the sibling MCP-token
    store so operators see symmetric pool usage across the two stores
    that usually share a DSN.
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._migrator = OrchidPostgresMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)
        logger.info(
            "[OrchidMCPGatewayStateStore:postgres] Initialised — %s",
            self._dsn.split("@")[-1] if "@" in self._dsn else "***",
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── Clients ──────────────────────────────────────────────

    async def register(self, record: OrchidMCPGatewayClient) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_gateway_clients
                    (client_id, client_name, redirect_uris, grant_types, response_types,
                     token_endpoint_auth_method, created_at)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6, $7)
                ON CONFLICT (client_id) DO UPDATE SET
                    client_name = EXCLUDED.client_name,
                    redirect_uris = EXCLUDED.redirect_uris,
                    grant_types = EXCLUDED.grant_types,
                    response_types = EXCLUDED.response_types,
                    token_endpoint_auth_method = EXCLUDED.token_endpoint_auth_method,
                    created_at = EXCLUDED.created_at
                """,
                record.client_id,
                record.client_name,
                json.dumps(record.redirect_uris),
                json.dumps(record.grant_types),
                json.dumps(record.response_types),
                record.token_endpoint_auth_method,
                float(record.created_at),
            )

    async def get(self, client_id: str) -> OrchidMCPGatewayClient | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_gateway_clients WHERE client_id = $1",
                client_id,
            )
        return _row_to_client(row) if row else None

    # ── Auth codes ───────────────────────────────────────────

    async def put(self, record: OrchidMCPGatewayAuthCode) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_gateway_auth_codes
                    (code, client_id, redirect_uri, code_challenge, code_challenge_method,
                     upstream_state, upstream_code_verifier, scopes, client_state, identity,
                     idp_access_token, idp_refresh_token, idp_expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10::jsonb,
                        $11, $12, $13, $14)
                """,
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
            )

    async def get_by_upstream_state(
        self,
        upstream_state: str,
    ) -> OrchidMCPGatewayAuthCode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_gateway_auth_codes WHERE upstream_state = $1",
                upstream_state,
            )
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
        sets: list[str] = []
        params: list[Any] = []
        idx = 1
        if identity is not None:
            sets.append(f"identity = ${idx}::jsonb")
            params.append(json.dumps(identity))
            idx += 1
        if idp_access_token is not None:
            sets.append(f"idp_access_token = ${idx}")
            params.append(idp_access_token)
            idx += 1
        if idp_refresh_token is not None:
            sets.append(f"idp_refresh_token = ${idx}")
            params.append(idp_refresh_token)
            idx += 1
        if idp_expires_at is not None:
            sets.append(f"idp_expires_at = ${idx}")
            params.append(float(idp_expires_at))
            idx += 1
        if not sets:
            return
        params.append(code)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE mcp_gateway_auth_codes SET {', '.join(sets)} WHERE code = ${idx}",
                *params,
            )

    async def consume(self, code: str) -> OrchidMCPGatewayAuthCode | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM mcp_gateway_auth_codes WHERE code = $1 RETURNING *",
                code,
            )
        return _row_to_auth_code(row) if row else None

    # ── Tokens ──────────────────────────────────────────────

    async def issue(self, record: OrchidMCPGatewayToken) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_gateway_tokens
                    (access_token, refresh_token, client_id, subject, identity,
                     scopes, expires_at,
                     idp_access_token, idp_refresh_token, idp_expires_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10)
                """,
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
            )

    async def get_by_access_token(
        self,
        access_token: str,
    ) -> OrchidMCPGatewayToken | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_gateway_tokens WHERE access_token = $1",
                access_token,
            )
        return _not_expired_token(_row_to_token(row) if row else None)

    async def get_by_refresh_token(
        self,
        refresh_token: str,
    ) -> OrchidMCPGatewayToken | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_gateway_tokens WHERE refresh_token = $1",
                refresh_token,
            )
        return _not_expired_token(_row_to_token(row) if row else None)

    async def revoke(self, access_token: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM mcp_gateway_tokens WHERE access_token = $1",
                access_token,
            )
        return result.endswith("1") if isinstance(result, str) else False


# ── Row mappers ─────────────────────────────────────────────


def _decode_jsonb(value: Any) -> Any:
    """asyncpg returns JSONB as Python objects already; fall back to
    ``json.loads`` if the driver handed us a string (belt-and-braces).
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_client(row: Any) -> OrchidMCPGatewayClient:
    return OrchidMCPGatewayClient(
        client_id=row["client_id"],
        client_name=row["client_name"],
        redirect_uris=_decode_jsonb(row["redirect_uris"]) or [],
        grant_types=_decode_jsonb(row["grant_types"]) or [],
        response_types=_decode_jsonb(row["response_types"]) or [],
        token_endpoint_auth_method=row["token_endpoint_auth_method"],
        created_at=float(row["created_at"]),
    )


def _row_to_auth_code(row: Any) -> OrchidMCPGatewayAuthCode:
    return OrchidMCPGatewayAuthCode(
        code=row["code"],
        client_id=row["client_id"],
        redirect_uri=row["redirect_uri"],
        code_challenge=row["code_challenge"],
        code_challenge_method=row["code_challenge_method"],
        upstream_state=row["upstream_state"],
        upstream_code_verifier=row["upstream_code_verifier"],
        scopes=_decode_jsonb(row["scopes"]) or [],
        client_state=row["client_state"],
        identity=_decode_jsonb(row["identity"]),
        idp_access_token=row["idp_access_token"],
        idp_refresh_token=row["idp_refresh_token"],
        idp_expires_at=float(row["idp_expires_at"]),
        created_at=float(row["created_at"]),
    )


def _row_to_token(row: Any) -> OrchidMCPGatewayToken:
    return OrchidMCPGatewayToken(
        access_token=row["access_token"],
        refresh_token=row["refresh_token"],
        client_id=row["client_id"],
        subject=row["subject"],
        identity=_decode_jsonb(row["identity"]) or {},
        scopes=_decode_jsonb(row["scopes"]) or [],
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
