# rag/ — Retrieval-Augmented Generation

## Overview

The RAG subsystem covers four pluggable axes, wired together by
hierarchical scoping and an additive metadata-filter mini-language:

| Axis              | ABC                          | Registry                     | Built-ins                                              |
| ----------------- | ---------------------------- | ---------------------------- | ------------------------------------------------------ |
| Ingestion         | `OrchidIngestionStrategy`    | `INGESTION_REGISTRY`         | `recursive`, `semantic`, `hierarchical`, `headered`    |
| Retrieval         | `OrchidRetrievalStrategy`    | `RETRIEVAL_REGISTRY`         | `simple`, `multi_query`, `hyde`, `hybrid`, `graph_rag` |
| Query transform   | `OrchidQueryTransformer`     | `TRANSFORMER_REGISTRY`       | `reformulate`, `multi_query`, `hyde`, `decompose`      |
| Sparse encoder    | `OrchidSparseEncoder`        | `SPARSE_ENCODER_REGISTRY`    | `bm25`; `splade` behind the optional `splade` extra    |

Concrete backends live behind their own registries
(`VECTOR_BACKEND_REGISTRY`, `DOC_STORE_BACKEND_REGISTRY`,
`GRAPH_STORE_BACKEND_REGISTRY`, `SPARSE_ENCODER_REGISTRY`) so swapping
Qdrant for OpenSearch — or the in-memory graph store for Neo4j — is a
config flip, never a code edit.

## Hierarchical Scope Model

Every RAG operation uses `OrchidRAGScope` — never raw filters.

```
"__shared__"                        ← visible to ALL tenants
└── tenant_id (str(installation_id))
    └── user_id (user_uuid)
        ├── scope="user"            ← visible across all user's chats
        └── chat_id
            ├── scope="chat_shared" ← visible to all agents in this chat
            └── scope="chat_agent"  ← private to one agent
                + agent_id
```

### Query: which data does a scope see?

`build_qdrant_filter(scope)` creates an OR filter. A query with `OrchidRAGScope(tenant_id="99999", user_id="dev-user", chat_id="abc")` sees:
1. All `__shared__` documents
2. All `tenant_id=99999, scope="tenant"` documents
3. All `tenant_id=99999, user_id="dev-user", scope="user"` documents
4. All `tenant_id=99999, user_id="dev-user", chat_id="abc", scope="chat_shared"` documents

### Write: how is scope assigned?

When indexing documents, set scope metadata explicitly:
- Static data (batch): `scope="tenant"` or `tenant_id="__shared__"`
- Tool results (dynamic): `scope="chat_shared"` (visible to all agents in chat)
- Uploaded files: `scope="chat_shared"` (chat-scoped by default)
- Shared data (promoted): `scope="user"` (after `POST /chats/{id}/share`)

## Metadata Filtering Mini-Language

`OrchidVectorReader.retrieve(...)` (and `retrieve_sparse(...)`) take an
optional `metadata_filters: dict[str, Any] | None` parameter that
operates **alongside** the scope filter — never replaces it. The
mini-language operators:

| Form                                  | Meaning                              |
| ------------------------------------- | ------------------------------------ |
| `{"status": "published"}`             | Exact match.                         |
| `{"language": ["en", "fr"]}`          | Match-any (OR within the field).     |
| `{"view_count": {"gte": 100}}`        | Range — `gte`/`lte`/`gt`/`lt`.        |
| `{"published_at": {"gte": "2026-01"}}`| ISO-8601 strings → `DatetimeRange`.   |
| `{"tags": {"contains": "alpha"}}`     | Substring / list-contains.           |
| `{"deprecated": {"not": True}}`       | Negation (`must_not`).               |
| `{"_qdrant": {...}}`                  | Backend-namespaced extras (skipped). |

The Qdrant backend translates these via
[`build_metadata_filter_clauses`](backends/qdrant.py) and infers
payload index types (keyword / integer / float / bool / datetime)
when the agent didn't declare `payload_indexes` explicitly.

## Per-Tool RAG Override

Each MCP tool (`mcp_servers[*].tools[*].rag`) and built-in tool
(`tools.<name>.rag`) may carry its own `rag:` block. At runtime
`OrchidAgentConfig.effective_rag(tool_name)` deep-merges
`tool.rag.model_dump(exclude_unset=True)` over `agent.rag.model_dump()`
— so a single tool can flip namespace, ingestion strategy, or chunk
size without restating the full block. `inject_to_rag` consumes the
result via `build_ingestion_strategy` so the tool's chunks land in
the configured layout, not a one-size-fits-all 2000-char truncation.

Use the **`exclude_dynamic: true`** retrieval flag to keep
dynamically-injected tool output out of the retrieval path — the
agent injects `dynamic: {"not": True}` into the metadata filters
automatically.

## Files

| File                                     | Purpose                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `scopes.py`                              | Re-exports `OrchidRAGScope` + `build_qdrant_filter()`                  |
| `dynamic.py`                             | `inject_to_rag()` — per-tool dynamic injection through a strategy      |
| `indexer.py`                             | `StaticIndexer` — batch indexes test/seed data at startup              |
| `embeddings.py`                          | `LiteLLMEmbedder` — generates embeddings via LiteLLM                   |
| `factory.py`                             | `build_reader()` / `build_doc_store()` / `build_graph_store()` / etc.  |
| `strategies/__init__.py`                 | `RETRIEVAL_REGISTRY` + `register_retrieval_strategy()`                 |
| `strategies/{simple,multi_query,hyde,hybrid,graph_rag}.py` | Built-in retrieval strategies                            |
| `transformers/__init__.py`               | `TRANSFORMER_REGISTRY` + `pre_strategy` flag handling                  |
| `transformers/{reformulate,multi_query,hyde,decompose}.py` | Built-in query transformers                              |
| `sparse/{__init__.py,bm25.py,splade.py}` | Sparse-encoder ABC + registry + built-ins                              |
| `backends/null.py`                       | `NullVectorReader` / `NullDocStore` / `NullGraphStore` (`is_null=True`) |
| `backends/qdrant.py`                     | `QdrantRepository` + filter translation                                |
| `backends/qdrant_doc_store.py`           | Qdrant-backed parent doc store                                         |
| `backends/in_memory_doc_store.py`        | Process-local parent doc store (tests / dev)                           |
| `backends/in_memory_graph.py`            | Process-local `OrchidGraphStore` impl                                  |
| `backends/neo4j_graph.py`                | Neo4j-backed `OrchidGraphStore` (optional `neo4j` extra)               |

## Key Rules

- **No concrete backend imports outside `backends/`.**
  Importing `qdrant_client`, `neo4j`, or any other client elsewhere
  fails the [`tests/test_dependency_boundaries.py`](../../tests/test_dependency_boundaries.py)
  architectural lint.
- **Embedding model dimensions are critical.** `nomic-embed-text` =
  768-d, `text-embedding-3-small` = 1536-d, `gemini-embedding-001` =
  3072-d. Switching models requires re-creating Qdrant collections.
- **`inject_to_rag()` is safe to call on a `NullVectorReader`** — it
  checks `isinstance(store, OrchidVectorWriter)` first and no-ops.
- **`inject_to_rag()` is per-tool, not per-agent.** Pass
  `tool_name`, `tool_result`, and a pre-built ingestion strategy.
  The agent constructs the strategy via
  `build_ingestion_strategy(effective_rag(tool_name).ingestion)`
  (`agents/generic_agent.py:_step_dynamic_injection`).
- **Strategies are stateless across calls.** Build once, re-use.
  Per-call resources (chat model, graph store, doc store, embedder)
  arrive as kwargs so the registry doesn't need to know which
  strategy needs which.
- **Pre-strategy transformers fire at agent entry, strategy-internal
  ones fan out inside the strategy.** The
  `pre_strategy: ClassVar[bool]` flag on each transformer drives the
  split.

## Adding a Custom Strategy

1. Subclass the relevant ABC (`OrchidRetrievalStrategy`,
   `OrchidIngestionStrategy`, `OrchidQueryTransformer`,
   `OrchidSparseEncoder`).
2. Implement the abstract methods plus an optional
   `from_config(config) -> Self` classmethod for YAML knobs.
3. Register at startup via `register_retrieval_strategy()` /
   `register_ingestion_strategy()` / `register_query_transformer()` /
   `register_sparse_encoder()`.
4. Reference by name in `agents.yaml` (`rag.retrieval.strategy`,
   `rag.ingestion.strategy`, etc.).

See [`examples/rag-strategies/`](../../../examples/rag-strategies/)
for a working `recency_simple` retrieval strategy registered from a
startup hook.

## QdrantRepository Notes

- Collections are created at startup via `ensure_collections()`.
- Hybrid collections carry a named dense vector + a sparse vector;
  `retrieve_sparse()` runs a sparse-only query that
  `HybridRetrieval` fuses with the dense lane via RRF (default) or
  weighted-linear.
- `duplicate_with_new_scope()` — copies points with new metadata
  (used by chat sharing).
- `scroll_by_scope()` — scrolls matching points (used to find
  chat-scoped data).
- Embedding happens lazily inside `upsert()` — documents without
  embeddings are embedded on the fly.
- Payload indexes — `OrchidRAGConfig.payload_indexes` declares
  explicit field types (`keyword` / `integer` / `float` / `bool` /
  `datetime`); when omitted, the backend infers from filter operands
  on first call.
