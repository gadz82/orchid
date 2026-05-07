# orchid/ — AI Context (Framework Library)

## What This Package Is

**orchid-ai** is the core Python library of the Orchid multi-agent AI framework. It is a pip-installable package (`from orchid_ai.xxx`) containing ABCs, `GenericAgent`, LangGraph graph builder, RAG pipeline, persistence, document parsing, and MCP client. It has **no API endpoints, no CLI, no vendor-specific code**. Those live in separate packages (`orchid-api/`, `orchid-cli/`) or consumer projects.

## Package Structure

```
orchid/
  orchid_ai/              Package root (import as `from orchid_ai.xxx`)
    __init__.py           SDK surface: OrchidAgent, OrchidAuthContext, build_graph, load_config, etc.
    core/                 Pure ABCs — ZERO external dependencies (only stdlib)
      agent.py            OrchidAgent ABC
      state.py            OrchidAuthContext + OrchidAgentState
      identity.py         OrchidIdentityResolver ABC (also powers /auth/resolve-identity)
      auth_config.py      OrchidAuthConfigProvider + OrchidAuthExchangeClient
                            (exchange + refresh ABCs and discovery shape)
      mcp.py              OrchidMCPToolCaller / OrchidMCPDiscoverable / OrchidMCPTokenStore ABCs
      mcp_gateway_state.py
                          OrchidMCPGatewayClientStore / …AuthCodeStore / …TokenStore ABCs
                            (inbound gateway DCR clients + auth codes + tokens)
      repository.py       OrchidVectorReader / OrchidVectorWriter / OrchidVectorStoreAdmin ABCs
    config/               YAML config loader + schema + tool registry (parameter metadata)
    agents/               GenericAgent + strategies (SkillDetector, MCPDispatcher, SkillExecutor)
    graph/                LangGraph wiring: supervisor.py, graph.py, state.py
    rag/                  Scopes, indexer, embeddings, factory, backends/qdrant.py
    documents/            Parsers (PDF/DOCX/XLSX/CSV/Image), chunker, pipeline
    persistence/          OrchidChatStorage + OrchidMCPTokenStore + gateway-state stores + shared migrations:
      sqlite.py                       OrchidSQLiteChatStorage (default, aiosqlite — core dep)
      postgres.py                     OrchidPostgresChatStorage (optional, asyncpg)
      mcp_token_sqlite.py             OrchidSQLiteMCPTokenStore (outbound MCP OAuth tokens)
      mcp_token_postgres.py           OrchidPostgresMCPTokenStore
      mcp_token_factory.py            build_mcp_token_store()
      mcp_client_registration_*.py    Per-server discovered endpoints + DCR creds (RFC 7591)
      mcp_gateway_state_sqlite.py     Inbound gateway-state store (SQLite)
      mcp_gateway_state_postgres.py   Inbound gateway-state store (Postgres)
      mcp_gateway_state_factory.py    build_mcp_gateway_state_store()
      migrations/                     Unified v001 — chat + outbound MCP + inbound gateway tables
    mcp/                  StreamableHttpMCPClient + OrchidMCPAuthRegistry
      client.py           StreamableHttpMCPClient (dual-mode: none/passthrough/oauth)
      auth_registry.py    OrchidMCPAuthRegistry — scans config for OAuth-requiring servers
    llm_factory.py        build_chat_model() — provider-first, ChatLiteLLM fallback
    utils.py              import_class() shared utility
  tests/                  720+ tests
  pyproject.toml
```

## Dependency Direction (MUST follow)

```
graph/ → agents/ → core/
         agents/ → rag/ → core/
         agents/ → mcp/ → core/
persistence/ → core/  (standalone)
documents/   → core/  (standalone)
```

`core/` depends only on `langchain-core` (for `Document` and message types). No concrete backend imports (`qdrant_client`, `asyncpg`, `litellm`) are allowed.

## Core ABCs (`orchid/core/`)

| ABC | File | Purpose |
|-----|------|---------|
| `OrchidAgent` | `agent.py` | Agent identity + `run()`, `summarise()`, `fetch_rag_context()`, `extract_user_query()`, `extract_conversation_history()` |
| `OrchidIdentityResolver` | `identity.py` | Bearer token → `OrchidAuthContext` (also drives `/auth/resolve-identity`) |
| `OrchidAuthConfigProvider` | `auth_config.py` | Resolves non-secret upstream-OAuth discovery for `/auth-info` |
| `OrchidAuthExchangeClient` | `auth_config.py` | Server-side authorization-code + refresh-token exchange — holds `client_secret` |
| `OrchidMCPToolCaller` | `mcp.py` | Call MCP tools |
| `OrchidMCPDiscoverable` | `mcp.py` | Discover MCP capabilities |
| `OrchidMCPTokenStore` | `mcp.py` | Per-user outbound OAuth token persistence |
| `OrchidMCPClientRegistrationStore` | `mcp.py` | Per-server discovered endpoints + DCR creds (RFC 7591) |
| `OrchidMCPGatewayClientStore` | `mcp_gateway_state.py` | Inbound DCR client registrations |
| `OrchidMCPGatewayAuthCodeStore` | `mcp_gateway_state.py` | Inbound in-flight auth codes |
| `OrchidMCPGatewayTokenStore` | `mcp_gateway_state.py` | Inbound issued access + refresh + IdP-token records |
| `OrchidVectorReader` | `repository.py` | Vector store retrieval |
| `OrchidVectorWriter` | `repository.py` | Vector store indexing |
| `OrchidVectorStoreAdmin` | `repository.py` | Collection management |
| `OrchidChatStorage` | `persistence/base.py` | Chat CRUD + message persistence |

**LLM abstraction:** Orchid uses LangChain's `BaseChatModel` directly (no custom ABC). Use `build_chat_model(model_string)` factory to create one from a LiteLLM-style model string.

**Document model:** Uses `langchain_core.documents.Document` (re-exported from `core/repository.py`). Fields: `page_content`, `metadata`, `id`.

**Embeddings:** Uses `langchain_core.embeddings.Embeddings`. Use `build_embeddings(model_string)` factory.

## Key Dependencies

| Package | Role | Required? |
|---------|------|-----------|
| langgraph | Agent graph framework | Core |
| langchain-core | ABCs: BaseChatModel, Embeddings, Document | Core |
| langchain-litellm | ChatLiteLLM fallback (wraps litellm) | Core |
| langchain-community | Community integrations | Core |
| langchain-text-splitters | RecursiveCharacterTextSplitter | Core |
| litellm | Multi-provider LLM routing (fallback) | Core |
| qdrant-client | Vector DB client | Core |
| aiosqlite | SQLite async driver (default storage) | Core |
| asyncpg | PostgreSQL async driver | Optional (`orchid-ai[postgres]`) |
| langchain-openai | OpenAI provider (optional, improves perf) | Optional |
| langchain-google-genai | Google AI provider (optional) | Optional |
| langchain-ollama | Ollama provider (optional) | Optional |
| mcp | MCP protocol client | Core |
| pymupdf | PDF parsing | Core |
| python-docx | DOCX parsing | Core |
| openpyxl | XLSX parsing | Core |

## Architecture Rules

1. **`orchid/core/` = ZERO external dependencies.** Only Python stdlib imports. Every other module depends on `core/`. Violating this is an architectural bug.

2. **No Qdrant imports outside `rag/backends/`.** All vector access goes through `OrchidVectorReader`/`OrchidVectorWriter`/`OrchidVectorStoreAdmin` ABCs in `core/repository.py`.

3. **Graph-level auth uses passthrough only.** The graph's `OrchidAuthContext` token is obtained ONCE at the API entry point. MCP servers with `auth.mode: passthrough` forward this token. MCP servers with `auth.mode: oauth` resolve their own per-user tokens from `OrchidMCPTokenStore`. MCP servers with `auth.mode: none` (default) send no auth headers.

4. **RAG always uses `OrchidRAGScope`.** Never pass raw `tenant_id` filters. 5-level hierarchy: root -> tenant -> user -> chat -> agent.

5. **Parse-once pattern for documents.** Call `extract_text()` once, pass to both prompt builder and `ingest_document(pre_extracted_text=...)`.

6. **Imports are `from orchid_ai.xxx`, not `from src.xxx`.** The three-package split uses `orchid_ai.` as the import root.

7. **No vendor-specific code — including in comments and docstrings.** Platform integrations belong in consumer projects. Code, comments, docstrings, and examples inside `orchid/orchid_ai/` must NEVER reference any concrete product, vendor name, or domain-specific object (e.g. specific business entities like "orders", "courses", "tickets" — unless used as a purely generic illustrative example that could apply to any integrator). Use domain-neutral placeholders (e.g. `knowledge-base`, `search`, `records`, `catalog`) when examples are unavoidable. Violations in comments are as bad as violations in code — they create false coupling and mislead future contributors.

8. **Consumer agents inherit from `OrchidAgent`** and use `self.summarise()`, `self.fetch_rag_context()`, `self.extract_user_query()`, `self.extract_conversation_history()` — don't duplicate these methods.

9. **Multi-turn conversation context is handled at framework level.** `OrchidAgent.extract_conversation_history()` extracts clean dialogue from graph state. `summarise()` accepts `conversation_history` and `prior_tool_context` parameters. The supervisor uses configurable `history_max_turns` (default 20) and `history_max_chars` (default 1000) from `OrchidSupervisorConfig`. Opt-in **sliding-window summarization** (`history_summary_enabled`) compresses older turns via a cheap LLM call, keeping the most recent `history_summary_recent_turns` (default 10) exchanges verbatim.

10. **MCP communication boundaries use broad exception handling.** `mcp_dispatcher.py` and `strategies.py` catch `Exception` (not a narrow tuple) at server/tool call boundaries. This is intentional fault isolation — MCP servers can fail with HTTP errors (401, 500), transport errors, or protocol errors, and one failing server must not crash the entire agent. Always use `except Exception` at these boundaries; never narrow it to a specific tuple.

11. **MCP servers support three auth modes** configured via `auth.mode` in `OrchidMCPServerConfig`: `none` (default — no auth headers, for local/unauthenticated servers), `passthrough` (forwards graph OrchidAuthContext bearer token), `oauth` (per-user tokens from OrchidMCPTokenStore with auto-refresh). The `OrchidMCPAuthRegistry` is built once at graph startup from `OrchidAgentsConfig` and exposes which servers require OAuth. `mcp_auth_status` is injected into graph state per-request so the supervisor can make auth-aware routing decisions.

12. **Built-in tool parameters are declared in YAML or auto-extracted.** The `tools:` section in `agents.yaml` supports an optional `parameters:` block per tool. When declared, YAML parameters take precedence. When omitted, parameters are auto-extracted from the Python function signature via `inspect`. Framework-injected params (`query`, `context`, `auth_context`, `**kwargs`) are filtered out automatically. Parameter metadata is used by the CLI skill generator to produce accurate documentation.

13. **Mini-agents are opt-in via `mini_agent.enabled: true`** on a top-level agent (no nesting). When enabled, a deterministic structured-output decomposer runs at the start of the agent's turn; if it returns `should_fork=True` the graph fans out into N parallel mini-agent nodes (default cap 3, hard cap 8) each running a focused agentic loop with a curated tool subset, then synthesises their outcomes back into one `AIMessage` via the aggregator. The decomposer hook lives at the **graph-wrapper** level (`graph._create_agent_node`), not inside `GenericAgent.run()` — so any `OrchidAgent` subclass can opt in via YAML without coordinating with its own `run()`. Cross-node data uses **shadow-slot keys**: `mini_agent_outcomes[f"{parent}#{mini_id}"]`, `mini_agent_decisions[parent_name]`. Four lifecycle SSE events (`mini_agent.{decomposed,started,finished,aggregated}`) surface activity to the streaming UI without leaking inner token streams. See the `mini_agent_*` modules under `orchid_ai/agents/`.

## Key Patterns

### Adding an Agent

**YAML only (most common):** Add entry to `agents.yaml`, `GenericAgent` handles everything.

**Custom class:** Subclass `OrchidAgent` in a consumer project, reference via dotted path in YAML:
```yaml
class: myproject.agents.custom.CustomAgent
```

### RAG Scoping

```python
from orchid_ai.rag.scopes import OrchidRAGScope

scope = OrchidRAGScope(
    tenant_id=auth.tenant_key,
    user_id=auth.user_uuid,
    chat_id=state.get("chat_id", ""),
    agent_id=self.name,
)
```

### OrchidRuntime (Dependency Bag)

```python
from orchid_ai import OrchidRuntime, build_graph, load_config

runtime = OrchidRuntime(
    default_model="gemini/gemini-2.5-flash",
    reader=my_qdrant_reader,           # or None → NullVectorReader
    chat_model=ChatOpenAI(model="gpt-4o"),  # or None → build_chat_model(default_model)
    mcp_client_factory=my_factory,      # or None → StreamableHttpMCPClient
)
graph = build_graph(config=load_config("agents.yaml"), runtime=runtime)
```

Integrators override only what they need. All fields have sensible defaults.

### Strategy Pattern (Tool Calls)

`all`, `sequential`, `llm_decides` are registered strategies. New ones: subclass `OrchidToolCallStrategy` + `register_strategy()`.

### LLM Usage

- **Simple completions** (summarization, routing): Use `self.summarise()` which calls `self._chat_model.ainvoke()`. A `BaseChatModel` must be injected via `chat_model=` — there is no fallback.
- **Agentic tool-calling loops** (need `tool_calls` response): Use `litellm` directly with lazy import inside the method. Add a comment explaining why.
- **Never import `litellm` at module level** in consumer agents.

## Testing

```bash
cd orchid && source .venv/bin/activate
pytest tests/ -x              # all tests (720+)
pytest tests/ -k "test_scopes"  # specific
ruff check orchid_ai/         # lint
ruff format orchid_ai/        # format
```

## Embedding Dimensions (Critical for Qdrant)

| Model | Dimensions |
|-------|-----------|
| ollama/nomic-embed-text | 768 |
| text-embedding-3-small | 1536 |
| gemini/gemini-embedding-001 | 3072 |

Switching models requires wiping and re-indexing Qdrant collections.

## MCP gateway exposure config (optional)

`OrchidAgentsConfig.mcp_gateway` lets integrators customise how Orchid's
MCP-facing gateway (e.g. `orchid-mcp`) presents itself to host LLMs:

- **Tool overrides** — replace default `title` / `description` for
  specific tool names.
- **MCP Prompts** — pre-canned templates with `{{var}}` substitution.

The block is **entirely optional** — a YAML without `mcp_gateway:`
parses into an empty config (`default_factory`), and nothing in
`orchid/` depends on it being populated. Data-only: the framework
library does not render templates or track which tools a gateway
actually exposes (platform-agnostic, keeps the hard-rule boundary).

```yaml
mcp_gateway:
  tools:
    orchid_ask:
      title: "Ask the Acme Knowledge Base"
      description: "Route a question to the Acme support agents."
    # The Pollen + Bloom tools ship via the same surface — any
    # tool name registered by the gateway can be re-titled and
    # re-described here without a gateway code change.
    orchid_signal_emit:
      title: "Trigger a background workflow"
      description: "Emit a Pollen signal to start an event-driven Bloom run."
    orchid_bloom_status:
      title: "Check background work status"
      description: "Look up the status of a Bloom run."
    orchid_bloom_list:
      title: "List recent background runs"
      description: "Browse Bloom runs with optional status / trigger filters."
  prompts:
    - name: compliance_report
      description: "Generate a compliance-completion report."
      arguments:
        - { name: department, required: true }
      template: |
        Produce a compliance report for {{department}}.
```

Tool keys are arbitrary — the schema accepts any tool name the
configured gateway exposes (built-in `orchid_*` tools, plus any
custom tools an integrator wires up). Add an entry and the next
MCP session sees the override on its first `tools/list`. The
default titles/descriptions live in the gateway's source
(`orchid-mcp/src/tools/`); a missing override keeps those defaults.

Classes live in `orchid_ai/config/mcp_gateway.py`. Env-var overrides +
external prompts-file loading happen upstream in `orchid-api`'s
`mcp_gateway.py`; the framework library has no env-var logic.

## Auth-centralisation ABCs

The framework library carries the abstract auth surface; concrete
implementations live in consumer projects.

- `OrchidAuthConfigProvider` resolves the non-secret upstream-OAuth discovery shape (`OrchidUpstreamOAuthConfig`) consumed by `orchid-api`'s `/auth-info`. Pure: no network calls, no side effects; reads env vars seeded from `orchid.yml`.
- `OrchidAuthExchangeClient` holds the upstream `client_secret` and performs the authorization-code (`exchange_code`) and refresh-token (`refresh_token`) grants on behalf of downstream public PKCE clients. Default `refresh_token` raises `NotImplementedError` — exchange-only consumers don't break, and the `/auth-info` flag gating `refresh_via_api` checks the method override identity.
- Three `OrchidMCPGatewayClient/AuthCode/Token Store` ABCs back the inbound MCP gateway's OAuth state. One concrete class implements all three against the shared chat DB (SQLite / Postgres). `OrchidMCPGatewayToken` carries `idp_access_token` + `idp_refresh_token` + `idp_expires_at` so the refresh path has the upstream pair to swap.
- `OrchidIdentityResolver` (already pre-existing) does double-duty: per-request bearer validation AND the upstream-token → identity bridge exposed at `/auth/resolve-identity`.

## Common Pitfalls

- **Importing qdrant_client in agent code.** Use `self.reader.retrieve(...)` instead.
- **Forgetting `from __future__ import annotations`** at the top of new files.
- **Using `filters: dict` instead of `scope: OrchidRAGScope`** in retrieval calls.
- **Passing `tenant_id` directly** — use `auth.tenant_key` (which is `tenant_id or "default"`).
- **Not handling `_reader` being `None`** — vector backend can be `null` in tests.
- **Mutating `OrchidAuthContext`** — it's subclass-friendly but treat as immutable in framework code.
- **Adding API/CLI code here** — those belong in `orchid-api/` and `orchid-cli/`.
- **Catching only specific exceptions at MCP boundaries** — always use `except Exception` at server communication boundaries in `mcp_dispatcher.py` and `strategies.py`. HTTP libraries (httpx) raise exception types like `httpx.HTTPStatusError` (for 401/500) that are not subclasses of `ConnectionError`/`TimeoutError`/`OSError`. A narrow exception tuple lets these propagate and crash the agent.
- **Adding HTTP / fetch logic to `OrchidAuthConfigProvider`** — it's a pure config-resolution ABC. Network calls to validate the discovery block happen one layer up in `orchid-api`'s router.
