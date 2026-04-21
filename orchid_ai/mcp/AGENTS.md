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
- `OrchidOAuthStateStore` — ABC for persisting PKCE state between the
  `/authorize` redirect and the `/callback` exchange. `OrchidInMemoryOAuthStateStore`
  (default) is suitable for single-instance deployments; integrators running
  multiple replicas supply a Redis/DB backend via `orchid.yml`.

## Architecture

```
orchid_ai/mcp/
  client.py          StreamableHttpMCPClient — concrete HTTP-based client
  auth_registry.py   OrchidMCPAuthRegistry — which servers need OAuth + who's authorized
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

Configured in `agents.yaml` via `OrchidMCPServerConfig.auth`:

| Mode | Behaviour |
|------|-----------|
| `none` (default) | No auth headers. For local/unauthenticated servers. |
| `passthrough` | Forwards the graph's `OrchidAuthContext` bearer token on every MCP call. |
| `oauth` | Per-user token resolved from `OrchidMCPTokenStore`, auto-refreshed. |

```yaml
mcp_servers:
  - name: public-weather
    url: https://weather.example.com/mcp
    # auth defaults to: mode: none

  - name: tenant-catalog
    url: https://catalog.mytenant.example.com/mcp
    auth:
      mode: passthrough          # uses graph bearer token

  - name: google-drive
    url: https://drive-mcp.example.com/mcp
    auth:
      mode: oauth
      authorize_url: https://accounts.google.com/o/oauth2/v2/auth
      token_url: https://oauth2.googleapis.com/token
      scopes: ["https://www.googleapis.com/auth/drive.readonly"]
      client_id_env: GDRIVE_CLIENT_ID
      client_secret_env: GDRIVE_CLIENT_SECRET
```

## Per-request wiring (graph state)

`orchid-api` injects `mcp_auth_status: dict[str, bool]` into the graph state
at the start of every request (one key per OAuth-requiring server). The
supervisor reads this to decide whether to dispatch an agent that depends
on an unauthorised server, or return an `auth_required` response to the
client instead.

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
- Hardcoding OAuth client secrets in `agents.yaml`. Use the `*_env`
  suffixed keys (e.g. `client_secret_env: GDRIVE_CLIENT_SECRET`) to
  resolve from the process environment instead.
