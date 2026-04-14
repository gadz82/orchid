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
      llm_provider.py     LLMProvider ABC
      mcp.py              MCPToolCaller / MCPDiscoverable ABCs
      repository.py       VectorReader / VectorWriter / VectorStoreAdmin ABCs
    config/               YAML config loader + schema + registries
    agents/               GenericAgent + strategies (SkillDetector, MCPDispatcher, SkillExecutor)
    graph/                LangGraph wiring: supervisor.py, graph.py, state.py
    rag/                  Scopes, indexer, embeddings, factory, backends/qdrant.py
    documents/            Parsers (PDF/DOCX/XLSX/CSV/Image), chunker, pipeline
    persistence/          ChatStorage ABC + factory + models + migrations + built-in backends:
      sqlite.py           SQLiteChatStorage (default, aiosqlite — core dep)
      postgres.py         PostgresChatStorage (optional, asyncpg — `pip install orchid-ai[postgres]`)
    mcp/                  StreamableHttpMCPClient
    llm_service.py        LiteLLMProvider (concrete LLMProvider)
    utils.py              import_class() shared utility
  tests/                  328+ tests
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

`core/` is the leaf — depends on NOTHING external. Never add `import qdrant_client`, `import asyncpg`, `import litellm`, or any third-party library to files in `core/`.

## Core ABCs (`orchid/core/`)

| ABC | File | Purpose |
|-----|------|---------|
| `BaseAgent` | `agent.py` | Agent identity + `run()`, `summarise()`, `fetch_rag_context()`, `extract_user_query()`, `extract_conversation_history()` |
| `IdentityResolver` | `identity.py` | Bearer token -> AuthContext |
| `LLMProvider` | `llm_provider.py` | Abstract LLM completion (`complete()`) |
| `MCPToolCaller` | `mcp.py` | Call MCP tools |
| `MCPDiscoverable` | `mcp.py` | Discover MCP capabilities |
| `VectorReader` | `repository.py` | Vector store retrieval |
| `VectorWriter` | `repository.py` | Vector store indexing |
| `VectorStoreAdmin` | `repository.py` | Collection management |
| `ChatStorage` | `persistence/base.py` | Chat CRUD + message persistence |

## Key Dependencies

| Package | Role | Required? |
|---------|------|-----------|
| langgraph | Agent graph framework | Core |
| litellm | Multi-provider LLM abstraction | Core |
| qdrant-client | Vector DB client | Core |
| aiosqlite | SQLite async driver (default storage) | Core |
| asyncpg | PostgreSQL async driver | Optional (`orchid-ai[postgres]`) |
| mcp | MCP protocol client | Core |
| pymupdf | PDF parsing | Core |
| python-docx | DOCX parsing | Core |
| openpyxl | XLSX parsing | Core |

## Architecture Rules

1. **`orchid/core/` = ZERO external dependencies.** Only Python stdlib imports. Every other module depends on `core/`. Violating this is an architectural bug.

2. **No Qdrant imports outside `rag/backends/`.** All vector access goes through `VectorReader`/`VectorWriter`/`VectorStoreAdmin` ABCs in `core/repository.py`.

3. **No OAuth in agents or MCP clients.** Token obtained ONCE at entry point, propagated via `AuthContext` (ADR-010).

4. **RAG always uses `RAGScope`.** Never pass raw `tenant_id` filters. 5-level hierarchy: root -> tenant -> user -> chat -> agent.

5. **Parse-once pattern for documents.** Call `extract_text()` once, pass to both prompt builder and `ingest_document(pre_extracted_text=...)`.

6. **Imports are `from orchid_ai.xxx`, not `from src.xxx`.** The three-package split uses `orchid_ai.` as the import root.

7. **No vendor-specific code.** Platform integrations belong in consumer projects.

8. **Consumer agents inherit from `BaseAgent`** and use `self.summarise()`, `self.fetch_rag_context()`, `self.extract_user_query()`, `self.extract_conversation_history()` — don't duplicate these methods.

9. **Multi-turn conversation context is handled at framework level.** `BaseAgent.extract_conversation_history()` extracts clean dialogue from graph state. `summarise()` accepts `conversation_history` and `prior_tool_context` parameters. The supervisor uses configurable `history_max_turns` (default 10) and `history_max_chars` (default 1000) from `SupervisorConfig`.

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
    llm_service=MyCustomProvider(),     # or None → LiteLLMProvider
    mcp_client_factory=my_factory,      # or None → StreamableHttpMCPClient
)
graph = build_graph(config=load_config("agents.yaml"), runtime=runtime)
```

Integrators override only what they need. All fields have sensible defaults. The old `build_graph(config=..., default_model=..., reader=...)` kwargs API still works (backward compat).

### Strategy Pattern (Tool Calls)

`all`, `sequential`, `llm_decides` are registered strategies. New ones: subclass `ToolCallStrategy` + `register_strategy()`.

### LLM Usage

- **Simple completions** (summarization, routing): Use `self.summarise()` or `self._llm_service` (routes through `LLMProvider` ABC). An `LLMProvider` **must** be injected — there is no litellm fallback.
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
