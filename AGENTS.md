# orchid/ — AI Context (Framework Library)

## What This Package Is

**orchid-ai** is the core Python library of the Orchid multi-agent AI framework. It is a pip-installable package (`from orchid_ai.xxx`) containing ABCs, `GenericAgent`, LangGraph graph builder, RAG pipeline, persistence, document parsing, and MCP client. It has **no API endpoints, no CLI, no vendor-specific code**. Those live in separate packages (`orchid-api/`, `orchid-cli/`) or consumer projects.

## Package Structure

```
orchid/
  orchid_ai/              Package root (import as `from orchid_ai.xxx`)
    __init__.py           SDK surface: BaseAgent, AuthContext, build_graph, load_config, etc.
    core/                 Pure ABCs — ZERO external dependencies (only stdlib)
      agent.py            BaseAgent ABC
      state.py            AuthContext + AgentState
      identity.py         IdentityResolver ABC
      llm_provider.py     (REMOVED — use BaseChatModel from langchain-core)
      mcp.py              MCPToolCaller / MCPDiscoverable ABCs
      repository.py       VectorReader / VectorWriter / VectorStoreAdmin ABCs
    config/               YAML config loader + schema + tool registry (parameter metadata)
    agents/               GenericAgent + strategies (SkillDetector, MCPDispatcher, SkillExecutor)
    graph/                LangGraph wiring: supervisor.py, graph.py, state.py
    rag/                  Scopes, indexer, embeddings, factory, backends/qdrant.py
    documents/            Parsers (PDF/DOCX/XLSX/CSV/Image), chunker, pipeline
    persistence/          ChatStorage + MCPTokenStore ABCs + factories + shared migrations:
      sqlite.py           SQLiteChatStorage (default, aiosqlite — core dep)
      postgres.py         PostgresChatStorage (optional, asyncpg — `pip install orchid-ai[postgres]`)
      mcp_token_sqlite.py SQLiteMCPTokenStore (per-server OAuth tokens, same DB)
      mcp_token_postgres.py PostgresMCPTokenStore (per-server OAuth tokens, same DB)
      mcp_token_factory.py  build_mcp_token_store() factory
      migrations/         Shared migrations (v001 = chat schema, v002 = token schema)
    mcp/                  StreamableHttpMCPClient + MCPAuthRegistry
      client.py           StreamableHttpMCPClient (dual-mode: none/passthrough/oauth)
      auth_registry.py    MCPAuthRegistry — scans config for OAuth-requiring servers
    llm_factory.py        build_chat_model() — provider-first, ChatLiteLLM fallback
    utils.py              import_class() shared utility
  tests/                  384+ tests
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
| `BaseAgent` | `agent.py` | Agent identity + `run()`, `summarise()`, `fetch_rag_context()`, `extract_user_query()`, `extract_conversation_history()` |
| `IdentityResolver` | `identity.py` | Bearer token -> AuthContext |
| `MCPToolCaller` | `mcp.py` | Call MCP tools |
| `MCPDiscoverable` | `mcp.py` | Discover MCP capabilities |
| `MCPTokenStore` | `mcp.py` | Per-server OAuth token persistence |
| `VectorReader` | `repository.py` | Vector store retrieval |
| `VectorWriter` | `repository.py` | Vector store indexing |
| `VectorStoreAdmin` | `repository.py` | Collection management |
| `ChatStorage` | `persistence/base.py` | Chat CRUD + message persistence |

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

2. **No Qdrant imports outside `rag/backends/`.** All vector access goes through `VectorReader`/`VectorWriter`/`VectorStoreAdmin` ABCs in `core/repository.py`.

3. **Graph-level auth uses passthrough only.** The graph's `AuthContext` token is obtained ONCE at the API entry point (ADR-010). MCP servers with `auth.mode: passthrough` forward this token. MCP servers with `auth.mode: oauth` resolve their own per-user tokens from `MCPTokenStore`. MCP servers with `auth.mode: none` (default) send no auth headers.

4. **RAG always uses `RAGScope`.** Never pass raw `tenant_id` filters. 5-level hierarchy: root -> tenant -> user -> chat -> agent.

5. **Parse-once pattern for documents.** Call `extract_text()` once, pass to both prompt builder and `ingest_document(pre_extracted_text=...)`.

6. **Imports are `from orchid_ai.xxx`, not `from src.xxx`.** The three-package split uses `orchid_ai.` as the import root.

7. **No vendor-specific code — including in comments and docstrings.** Platform integrations belong in consumer projects. Code, comments, docstrings, and examples inside `orchid/orchid_ai/` must NEVER reference specific consumer tools (course search, notifications, etc.), or any concrete product. Use generic terms ("tools requiring authenticated API access", not "any entity search, any entity management"). Violations in comments are as bad as violations in code — they create false coupling and mislead future contributors.

8. **Consumer agents inherit from `BaseAgent`** and use `self.summarise()`, `self.fetch_rag_context()`, `self.extract_user_query()`, `self.extract_conversation_history()` — don't duplicate these methods.

9. **Multi-turn conversation context is handled at framework level.** `BaseAgent.extract_conversation_history()` extracts clean dialogue from graph state. `summarise()` accepts `conversation_history` and `prior_tool_context` parameters. The supervisor uses configurable `history_max_turns` (default 20) and `history_max_chars` (default 1000) from `SupervisorConfig`. Opt-in **sliding-window summarization** (`history_summary_enabled`) compresses older turns via a cheap LLM call, keeping the most recent `history_summary_recent_turns` (default 10) exchanges verbatim.

10. **MCP communication boundaries use broad exception handling.** `mcp_dispatcher.py` and `strategies.py` catch `Exception` (not a narrow tuple) at server/tool call boundaries. This is intentional fault isolation — MCP servers can fail with HTTP errors (401, 500), transport errors, or protocol errors, and one failing server must not crash the entire agent. Always use `except Exception` at these boundaries; never narrow it to a specific tuple.

11. **MCP servers support three auth modes** configured via `auth.mode` in `MCPServerConfig`: `none` (default — no auth headers, for local/unauthenticated servers), `passthrough` (forwards graph AuthContext bearer token), `oauth` (per-user tokens from MCPTokenStore with auto-refresh). The `MCPAuthRegistry` is built once at graph startup from `AgentsConfig` and exposes which servers require OAuth. `mcp_auth_status` is injected into graph state per-request so the supervisor can make auth-aware routing decisions.

12. **Built-in tool parameters are declared in YAML or auto-extracted.** The `tools:` section in `agents.yaml` supports an optional `parameters:` block per tool. When declared, YAML parameters take precedence. When omitted, parameters are auto-extracted from the Python function signature via `inspect`. Framework-injected params (`query`, `context`, `auth_context`, `**kwargs`) are filtered out automatically. Parameter metadata is used by the CLI skill generator to produce accurate documentation.

## Key Patterns

### Adding an Agent

**YAML only (most common):** Add entry to `agents.yaml`, `GenericAgent` handles everything.

**Custom class:** Subclass `BaseAgent` in a consumer project, reference via dotted path in YAML:
```yaml
class: myproject.agents.custom.CustomAgent
```

### RAG Scoping

```python
from orchid_ai.rag.scopes import RAGScope

scope = RAGScope(
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

`all`, `sequential`, `llm_decides` are registered strategies. New ones: subclass `ToolCallStrategy` + `register_strategy()`.

### LLM Usage

- **Simple completions** (summarization, routing): Use `self.summarise()` which calls `self._chat_model.ainvoke()`. A `BaseChatModel` must be injected via `chat_model=` — there is no fallback.
- **Agentic tool-calling loops** (need `tool_calls` response): Use `litellm` directly with lazy import inside the method. Add a comment explaining why.
- **Never import `litellm` at module level** in consumer agents.

## Testing

```bash
cd orchid && source .venv/bin/activate
pytest tests/ -x              # all tests (328+)
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

## Common Pitfalls

- **Importing qdrant_client in agent code.** Use `self.reader.retrieve(...)` instead.
- **Forgetting `from __future__ import annotations`** at the top of new files.
- **Using `filters: dict` instead of `scope: RAGScope`** in retrieval calls.
- **Passing `tenant_id` directly** — use `auth.tenant_key` (which is `tenant_id or "default"`).
- **Not handling `_reader` being `None`** — vector backend can be `null` in tests.
- **Mutating `AuthContext`** — it's subclass-friendly but treat as immutable in framework code.
- **Adding API/CLI code here** — those belong in `orchid-api/` and `orchid-cli/`.
- **Catching only specific exceptions at MCP boundaries** — always use `except Exception` at server communication boundaries in `mcp_dispatcher.py` and `strategies.py`. HTTP libraries (httpx) raise exception types like `httpx.HTTPStatusError` (for 401/500) that are not subclasses of `ConnectionError`/`TimeoutError`/`OSError`. A narrow exception tuple lets these propagate and crash the agent.
