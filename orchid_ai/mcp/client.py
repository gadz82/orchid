"""
Concrete MCP client — supports Streamable HTTP and SSE transports.

Capabilities (tools, prompts, resources) live in a session-bounded
in-memory cache.  Discovery runs once on first use (or proactively via
:meth:`warm_cache`, driven by :class:`OrchidSessionWarmer`); the cache
survives for the lifetime of the process and is flushed only via
explicit :meth:`invalidate_cache` calls.  Only ``call_tool``,
``get_prompt``, and ``read_resource`` open fresh connections on every
invocation (they carry request-specific data).

Transport is selected via ``transport`` parameter:
  - ``"streamable_http"`` (default) — modern MCP protocol, uses POST
  - ``"sse"`` — legacy MCP protocol, uses GET for SSE stream + POST for messages
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from ..core.mcp import OrchidCacheableMCPClient, OrchidMCPToolResult
from ..core.state import OrchidAuthContext

logger = logging.getLogger(__name__)


@dataclass
class _CapabilitiesCache:
    """In-memory cache for MCP server capabilities."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    resource_contents: dict[str, str] = field(default_factory=dict)
    rendered_prompts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: Negative cache for resources whose initial ``read_resource`` call
    #: failed (server-side ENOENT, transport timeout, …).  Subsequent
    #: ``read_resource`` calls return an empty string instead of paying
    #: the round-trip cost again — the failure is permanent until
    #: :meth:`invalidate_cache` is called.
    failed_resource_uris: set[str] = field(default_factory=set)
    populated: bool = False


class StreamableHttpMCPClient(OrchidCacheableMCPClient):
    """MCP client that connects via Streamable HTTP or SSE transport.

    Capabilities are discovered once on first use (or on
    :meth:`warm_cache`) and live for the lifetime of the process —
    only :meth:`invalidate_cache` flushes the cache.  Use
    :class:`orchid_ai.mcp.session_warmer.OrchidSessionWarmer` to
    populate caches at the right lifecycle boundary so the per-request
    hot path stops paying the discovery cost.

    When ``auth_mode`` is ``"oauth"``, the client resolves per-user
    tokens from the ``token_store`` instead of using the graph's
    ``OrchidAuthContext`` bearer token (passthrough).
    """

    def __init__(
        self,
        url: str,
        server_type: str = "local",
        transport: str = "streamable_http",
        *,
        server_name: str = "",
        auth_mode: str = "passthrough",
        token_store: Any | None = None,  # OrchidMCPTokenStore (lazy import to avoid circular)
        registration_store: Any | None = None,  # OrchidMCPClientRegistrationStore
    ) -> None:
        self._url = url
        self._server_type = server_type
        self._transport = transport
        self._cache = _CapabilitiesCache()
        # Per-server OAuth fields
        self._server_name = server_name or url
        self._auth_mode = auth_mode
        self._token_store = token_store
        #: Per-server discovered metadata + DCR credentials — consulted
        #: on every oauth-mode token refresh so we use the right
        #: endpoints and auth method without anything in YAML.
        self._registration_store = registration_store

    @property
    def server_url(self) -> str:
        return self._url

    # ── Cache management ─────────────────────────────────────

    def _cache_valid(self) -> bool:
        """Return True if the cache has been populated.

        The cache lives for the lifetime of the process — it is flushed
        only via :meth:`invalidate_cache` (or a new admin endpoint).
        """
        return self._cache.populated

    def invalidate_cache(self) -> None:
        """Force re-discovery on next call."""
        self._cache = _CapabilitiesCache()

    async def warm_cache(self, auth: OrchidAuthContext) -> None:
        """
        Pre-populate the capabilities cache.

        Call this during startup or first interaction to avoid latency later.
        """
        if self._cache_valid():
            return
        await self._discover_and_cache(auth)

    async def _discover_and_cache(self, auth: OrchidAuthContext) -> None:
        """Discover all capabilities in a single connection and populate the cache."""
        logger.info("[MCP] Discovering capabilities from %s …", self._url)

        async with self._connect(auth) as session:
            # Tools
            try:
                tools_result = await session.list_tools()
                self._cache.tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "schema": t.inputSchema,
                        # MCP 2025-03-26 ``Tool.annotations`` — advisory hints
                        # (readOnlyHint / idempotentHint / destructiveHint /
                        # openWorldHint).  Stored as raw because downstream
                        # parsing (``MCPToolAnnotations.from_raw``) handles
                        # both dict and object shapes — this keeps the cache
                        # transport-agnostic.  ``None`` when the server omits
                        # the field.
                        "annotations": getattr(t, "annotations", None),
                    }
                    for t in tools_result.tools
                ]
                logger.info("[MCP] Cached %d tools from %s", len(self._cache.tools), self._url)
            except Exception as exc:
                logger.warning("[MCP] list_tools failed for %s: %s", self._url, exc)

            # Prompts
            try:
                prompts_result = await session.list_prompts()
                self._cache.prompts = [
                    {
                        "name": p.name,
                        "description": p.description or "",
                        "arguments": [
                            {"name": a.name, "description": a.description or "", "required": a.required or False}
                            for a in (p.arguments or [])
                        ],
                    }
                    for p in prompts_result.prompts
                ]
                logger.info("[MCP] Cached %d prompts from %s", len(self._cache.prompts), self._url)

                # Pre-render prompts that have NO required arguments
                for prompt_def in self._cache.prompts:
                    has_required = any(a.get("required") for a in prompt_def.get("arguments", []))
                    if has_required:
                        continue
                    try:
                        result = await session.get_prompt(prompt_def["name"], {})
                        self._cache.rendered_prompts[prompt_def["name"]] = [
                            {
                                "role": m.role,
                                "content": getattr(m.content, "text", str(m.content)),
                            }
                            for m in result.messages
                        ]
                    except Exception as exc:
                        logger.warning("[MCP] Could not pre-render prompt '%s': %s", prompt_def["name"], exc)
            except Exception as exc:
                logger.warning("[MCP] list_prompts failed for %s: %s", self._url, exc)

            # Resources
            try:
                resources_result = await session.list_resources()
                self._cache.resources = [
                    {
                        "uri": str(r.uri),
                        "name": r.name,
                        "description": r.description or "",
                        "mime_type": r.mimeType or "",
                    }
                    for r in resources_result.resources
                ]
                logger.info("[MCP] Cached %d resources from %s", len(self._cache.resources), self._url)

                # Pre-read text resources
                for res in self._cache.resources:
                    try:
                        read_result = await session.read_resource(res["uri"])
                        parts: list[str] = []
                        for content in read_result.contents:
                            if hasattr(content, "text"):
                                parts.append(content.text)
                            elif hasattr(content, "blob"):
                                parts.append(f"[binary data, {len(content.blob)} bytes]")
                        self._cache.resource_contents[res["name"]] = "\n".join(parts)
                    except Exception as exc:
                        # Negative-cache the failed URI so subsequent
                        # ``read_resource`` calls don't pay the
                        # round-trip cost again.  The failure is
                        # permanent until ``invalidate_cache`` is called.
                        self._cache.failed_resource_uris.add(res["uri"])
                        logger.warning("[MCP] Could not pre-read resource '%s': %s", res["uri"], exc)
            except Exception as exc:
                logger.warning("[MCP] list_resources failed for %s: %s", self._url, exc)

        self._cache.populated = True
        logger.info(
            "[MCP] Cache populated for %s: %d tools, %d prompts (%d rendered), %d resources (%d read)",
            self._url,
            len(self._cache.tools),
            len(self._cache.prompts),
            len(self._cache.rendered_prompts),
            len(self._cache.resources),
            len(self._cache.resource_contents),
        )

    # ── Auth ─────────────────────────────────────────────────

    async def _resolve_auth_headers(self, auth: OrchidAuthContext) -> dict[str, str]:
        """Resolve the Authorization header based on auth mode.

        - **none** (default): no auth headers — suitable for local or
          unauthenticated MCP servers.
        - **passthrough**: forwards the graph's ``OrchidAuthContext`` bearer
          token unchanged — zero overhead.
        - **oauth**: looks up the per-user token from the token store,
          auto-refreshes if expired, or raises ``OrchidMCPAuthRequiredError``
          when the user has not authorized this server.
        """
        if self._auth_mode == "none":
            return {}

        if self._auth_mode == "passthrough":
            return auth.bearer_header

        from ..core.mcp import OrchidMCPAuthRequiredError, OrchidMCPTokenRecord

        if not self._token_store:
            raise OrchidMCPAuthRequiredError(self._server_name)

        record: OrchidMCPTokenRecord | None = await self._token_store.get_token(
            auth.tenant_key,
            auth.user_id,
            self._server_name,
        )

        if record is None:
            raise OrchidMCPAuthRequiredError(self._server_name)

        # Auto-refresh expired tokens when a refresh_token is available
        if record.is_expired and record.is_refresh_available:
            record = await self._refresh_oauth_token(record)

        if record.is_expired:
            raise OrchidMCPAuthRequiredError(self._server_name)

        return record.bearer_header

    async def _refresh_oauth_token(self, record: Any) -> Any:
        """Exchange a refresh token for a new access token.

        Endpoints + client credentials are fetched from the
        :class:`OrchidMCPClientRegistrationStore` row that discovery
        populated — nothing is read from YAML for oauth-mode refresh.
        On failure, returns the stale record unchanged; the caller
        checks ``is_expired`` and raises ``OrchidMCPAuthRequiredError``.
        """
        import time as _time

        if self._registration_store is None:
            logger.warning(
                "[MCP] No registration_store for '%s' — cannot refresh oauth token",
                self._server_name,
            )
            return record

        registration = await self._registration_store.get(self._server_name)
        if registration is None or not registration.token_endpoint:
            logger.warning(
                "[MCP] No stored registration for '%s' — cannot refresh",
                self._server_name,
            )
            return record

        try:
            import httpx

            # Send client credentials via the request body
            # ("client_secret_post").  Discovery records the advertised
            # ``token_endpoint_auth_methods_supported`` on the
            # registration; when the authorization server declares Basic
            # we respect that, otherwise we default to body-form —
            # duplicating ``client_id`` across Basic + body breaks
            # strict OAuth2 servers with ``invalid_request``.
            request_data: dict[str, Any] = {
                "grant_type": "refresh_token",
                "refresh_token": record.refresh_token,
                "client_id": registration.client_id,
            }
            basic_auth = None
            if registration.client_secret:
                if registration.uses_basic_auth:
                    basic_auth = (registration.client_id, registration.client_secret)
                else:
                    request_data["client_secret"] = registration.client_secret

            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.post(
                    registration.token_endpoint,
                    data=request_data,
                    auth=basic_auth,
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "[MCP] Refresh rejected by '%s' (%d): %s",
                        registration.token_endpoint,
                        resp.status_code,
                        resp.text[:500],
                    )
                    return record
                data = resp.json()

            from ..core.mcp import OrchidMCPTokenRecord

            new_record = OrchidMCPTokenRecord(
                server_name=record.server_name,
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", record.refresh_token),
                expires_at=_time.time() + data.get("expires_in", 3600),
                scopes=record.scopes,
                created_at=record.created_at,
                updated_at=_time.time(),
            )
            await self._token_store.save_token(new_record)
            logger.info("[MCP] Refreshed OAuth token for server '%s'", self._server_name)
            return new_record

        except Exception as exc:
            logger.warning("[MCP] Token refresh failed for '%s': %s", self._server_name, exc)
            return record

    # ── Transport ────────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self, auth: OrchidAuthContext, *, timeout: float = 30.0) -> AsyncGenerator[ClientSession, None]:
        """Open a transport connection and yield an initialized ClientSession."""
        import asyncio

        headers = await self._resolve_auth_headers(auth)

        async with asyncio.timeout(timeout):
            if self._transport == "sse":
                async with sse_client(self._url, headers=headers) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
            else:
                async with streamablehttp_client(self._url, headers=headers) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session

    # ── Tool execution (always live — not cached) ────────────

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: OrchidAuthContext,
        *,
        timeout: float = 60.0,
    ) -> OrchidMCPToolResult:
        """Invoke a named tool on the MCP server."""
        logger.info(
            "[MCP] call_tool → %s | params: %s | url: %s [%s]",
            tool_name,
            arguments,
            self._url,
            self._transport,
        )

        async with self._connect(auth, timeout=timeout) as session:
            result = await session.call_tool(tool_name, arguments)
            content = [{"type": c.type, "text": getattr(c, "text", "")} for c in result.content]

            result_obj = OrchidMCPToolResult(
                content=content,
                is_error=result.isError or False,
            )

            logger.info(
                "[MCP] call_tool ← %s | is_error: %s | response: %s",
                tool_name,
                result_obj.is_error,
                result_obj.text[:500] if len(result_obj.text) > 500 else result_obj.text,
            )

            return result_obj

    # ── Cached discovery methods ─────────────────────────────

    async def list_tools(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List all tools (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.tools

    async def list_prompts(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List all prompts (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.prompts

    async def list_resources(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List all resources (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.resources

    # ── Prompt / Resource access ─────────────────────────────

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: OrchidAuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template (cached for zero-arg prompts)."""
        # Return from cache if pre-rendered and no arguments provided
        if not arguments and name in self._cache.rendered_prompts:
            return self._cache.rendered_prompts[name]

        async with self._connect(auth) as session:
            result = await session.get_prompt(name, arguments)
            return [
                {
                    "role": m.role,
                    "content": getattr(m.content, "text", str(m.content)),
                }
                for m in result.messages
            ]

    async def read_resource(self, uri: str, auth: OrchidAuthContext) -> str:
        """Read resource content (cached if previously discovered).

        Resources whose initial pre-read failed are negative-cached: a
        repeat call returns an empty string immediately rather than
        re-attempting a network round-trip every request.  The negative
        cache is cleared by :meth:`invalidate_cache`.
        """
        # Negative cache short-circuit — avoid retrying known-failed URIs
        if uri in self._cache.failed_resource_uris:
            logger.debug("[MCP] Resource '%s' previously failed; returning empty (negative cache)", uri)
            return ""

        # Check cache by URI or name
        for res_name, content in self._cache.resource_contents.items():
            if res_name == uri:
                return content
        # Also check by matching URI in the resource list
        for res in self._cache.resources:
            if res["uri"] == uri and res["name"] in self._cache.resource_contents:
                return self._cache.resource_contents[res["name"]]

        # Not cached — fetch live, negative-caching on failure
        try:
            async with self._connect(auth) as session:
                result = await session.read_resource(uri)
                parts: list[str] = []
                for content in result.contents:
                    if hasattr(content, "text"):
                        parts.append(content.text)
                    elif hasattr(content, "blob"):
                        parts.append(f"[binary data, {len(content.blob)} bytes]")
                return "\n".join(parts)
        except Exception:
            self._cache.failed_resource_uris.add(uri)
            raise
