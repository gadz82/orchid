# mcp/ — MCP Client + Per-Server OAuth

## Overview

Everything the framework needs to talk to [Model Context Protocol](https://modelcontextprotocol.io/)
servers over HTTP:

- `StreamableHttpMCPClient` — concrete client combining `OrchidMCPToolCaller`
  + `OrchidMCPDiscoverable`. Supports three auth modes per-server: `none`,
  `passthrough`, `oauth`.
- `OrchidMCPAuthRegistry` — computed once at graph startup from the loaded
  `OrchidAgentsConfig`; tells the supervisor which servers require per-user
  OAuth and exposes current auth status for each user+server pair.
- `OrchidMCPServerInventory` — broader cousin of the auth registry that
  carries **every** declared MCP server (not just OAuth ones). Built once
  from `OrchidAgentsConfig` and consumed by the session warmer.
- `OrchidSessionWarmer` — drives `OrchidMCPClient.warm_cache(auth)` at the
  right lifecycle boundaries: process startup for `auth.mode: none` servers,
  user-session start for `passthrough`/`oauth` servers. Idempotent per
  `(tenant_key, user_id)`.
- `OrchidOAuthStateStore` — ABC for persisting PKCE state between the
  `/authorize` redirect and the `/callback` exchange. `OrchidInMemoryOAuthStateStore`
  (default) is suitable for single-instance deployments; integrators running
  multiple replicas supply a Redis/DB backend via `orchid.yml`.

## Architecture

```
orchid_ai/mcp/
  client.py          StreamableHttpMCPClient — concrete HTTP-based client
                     (session-bounded capability cache, no TTL)
  auth_registry.py   OrchidMCPAuthRegistry — which servers need OAuth + who's authorized
  inventory.py       OrchidMCPServerInventory — every declared server, deduped by
                     (server_name, url), with mode + agent_names; clients_for(entry,
                     agents) returns the per-agent OrchidMCPClient instances bound to it
  session_warmer.py  OrchidSessionWarmer — proactive warm-up driving
                     OrchidMCPClient.warm_cache(auth) at startup / session start;
                     OrchidWarmReport carries warmed/skipped/failed
  oauth_state.py     OrchidOAuthStateStore ABC + OrchidInMemoryOAuthStateStore
                     + build_oauth_state_store() factory
                     + register_oauth_state_store() for custom backends
```

## ABCs

| ABC | Role | Subclass target? |
|-----|------|------------------|
| `OrchidMCPToolCaller` | Invoke a tool on an MCP server | yes (alternative transports) |
| `OrchidMCPDiscoverable` | List tools/prompts/resources | yes |
| `OrchidMCPClient` | Combines `ToolCaller` + `Discoverable` | yes |
| `OrchidMCPTokenStore` | Per-server OAuth token persistence | yes (e.g. Redis backend) |
| `OrchidOAuthStateStore` | PKCE state ↔ redirect state | yes (e.g. Redis) |

All ABCs live in `orchid_ai/core/mcp.py` and `orchid_ai/mcp/oauth_state.py`.

## Auth modes (per-server)

Configured in `agents.yaml` via `OrchidMCPServerConfig.auth`. Only `mode`
is carried in YAML — everything else is either handled by the graph
(`passthrough`) or discovered at runtime (`oauth`).

| Mode | Behaviour |
|------|-----------|
| `none` (default) | No auth headers. For local/unauthenticated servers. |
| `passthrough` | Forwards the graph's `OrchidAuthContext` bearer token on every MCP call. |
| `oauth` | **MCP 2025-03-26 flow.** On the first 401 the framework runs RFC 9728 → RFC 8414 → RFC 7591 discovery and dynamically registers a client. Per-user tokens are stored in `OrchidMCPTokenStore`, per-server endpoints + DCR credentials in `OrchidMCPClientRegistrationStore`. |

```yaml
mcp_servers:
  - name: public-weather
    url: https://weather.example.com/mcp
    # auth defaults to: mode: none

  - name: tenant-catalog
    url: https://catalog.mytenant.example.com/mcp
    auth:
      mode: passthrough          # uses graph bearer token

  - name: external-crm
    url: https://crm.example.com/mcp
    auth:
      mode: oauth                # that's it — framework handles the rest
```

## MCP 2025-03-26 authorization flow (what happens on `mode: oauth`)

1. First tool call hits the MCP server with no auth.
2. Server returns `401 + WWW-Authenticate: Bearer resource_metadata="…"`.
3. Framework (via `OrchidMCPAuthDiscovery` in `discovery.py`) fetches:
   - **Protected Resource Metadata** (RFC 9728) — names the auth server(s).
   - **Authorization Server Metadata** (RFC 8414) — endpoints + supported
     auth methods / grant types / PKCE / scopes.
   - **Dynamic Client Registration** (RFC 7591) — POST's client metadata
     (redirect URI + supported grants) and receives a fresh `client_id`
     (+ optional `client_secret`) for that server.
4. The resulting `OrchidMCPClientRegistration` is persisted — one row per
   MCP server — so every subsequent container lifetime reuses the same
   registration instead of creating a new one on each boot.
5. `OrchidMCPAuthRequiredError` is raised to the agent boundary; the
   API's `/mcp/auth/*` router drives the browser half of the dance
   (authorization URL + PKCE callback + token storage).
6. Subsequent user turns resolve the per-user token from
   `OrchidMCPTokenStore` and auto-refresh against the discovered token
   endpoint using the stored credentials.

The authorization server MUST advertise `registration_endpoint` in its
RFC 8414 metadata. If it doesn't, discovery fails with an
`OrchidMCPDiscoveryError` naming the missing piece. Integrators whose
IdP lacks DCR pre-seed `OrchidMCPClientRegistrationStore` with the
relevant endpoints + credentials before first use.

## Per-request wiring (graph state)

`orchid-api` injects `mcp_auth_status: dict[str, bool]` into the graph state
at the start of every request (one key per OAuth-requiring server). The
supervisor reads this to decide whether to dispatch an agent that depends
on an unauthorised server, or return an `auth_required` response to the
client instead.

## Capability cache lifecycle

`StreamableHttpMCPClient` keeps an in-memory `_CapabilitiesCache` that
holds discovered tools / prompts / resources / pre-rendered prompts /
pre-read resource bodies for the **lifetime of the process**. The
legacy 5-minute TTL is gone — capabilities are flushed only via:

- `client.invalidate_cache()` (per-server),
- `OrchidSessionWarmer.invalidate_user(auth)` (drops every passthrough/
  oauth client cache for that user pair AND the warmed-user record),
- `OrchidSessionWarmer.invalidate_server(server_name)`.

Caches are populated proactively by `OrchidSessionWarmer`:

| Boundary | Trigger | What gets warmed |
|----------|---------|------------------|
| Process startup | `Orchid.warm_unauthenticated_capabilities()` from `orchid-api` lifespan / `orchid-cli` bootstrap | every `auth.mode: none` server |
| User session start | `OrchidSessionWarmer.warm_for_user(auth)` from `POST /session/warm` (or the lazy backstop on first authenticated request) | every `passthrough` + `oauth` server |
| Per-server post-OAuth | `warm_one_for_user(auth, server_name)` from `oauth_callback` | the freshly-authorized server |

The supervisor / agentic loop hot path stops issuing discovery RPCs
once the cache is populated — `MCPDispatcher.render_capabilities` uses
the cached result from each `OrchidMCPClient.list_*` call. The cache
TTL is gone; manual invalidation is the only way to flush.

The `OrchidMCPServerConfig.cache_ttl` YAML field is rejected by
Pydantic (`extra="forbid"`); the `StreamableHttpMCPClient` constructor
no longer accepts a `cache_ttl=` kwarg.

## OAuth state store

Multi-worker deployments need a shared PKCE/CSRF state store so the
`/authorize` and `/callback` legs can hit different replicas. Default is
in-memory (single-instance); swap for a persistent backend via
`orchid.yml`:

```yaml
# single instance (default):
oauth_state_store_class: memory

# custom class (any dotted path to a OrchidOAuthStateStore subclass):
oauth_state_store_class: myapp.oauth.redis.RedisStateStore
oauth_state_store_dsn: redis://localhost:6379/0
oauth_state_ttl_seconds: 600
```

Register additional built-in types via `register_oauth_state_store("redis",
RedisStateStore)` at startup — the factory then accepts the short name.

## Key patterns

- Token lookup in the graph: `self._runtime.mcp_token_store.get_token(...)`
  — never reach past the ABC.
- Catch `OrchidMCPAuthRequiredError` at the agent boundary and propagate
  to the supervisor; don't log it as a generic error.
- New transports (stdio, WebSocket) belong in new concrete classes alongside
  `StreamableHttpMCPClient`, not by modifying it.

## Common pitfalls

- Forgetting to add an MCP server to `agents.yaml`'s `defaults.mcp_servers`
  or per-agent `mcp_servers` — `OrchidMCPAuthRegistry` won't know about it
  and OAuth flows won't initiate.
- Using `passthrough` auth for a server that requires per-user OAuth. The
  graph's bearer token is a single-identity token; per-user APIs need
  `oauth` mode.
- Putting client credentials or endpoint URLs in `agents.yaml`. The
  MCP 2025-03-26 schema accepts only `mode:` — anything else is either
  rejected by Pydantic or silently ignored. Endpoints + credentials
  live in the discovered `OrchidMCPClientRegistration` row.
- Pointing at an authorization server that doesn't expose RFC 7591
  dynamic client registration. Discovery fails loudly with an
  `OrchidMCPDiscoveryError`. For IdPs without DCR, seed
  `OrchidMCPClientRegistrationStore` manually before first use.
