# rag/ — Retrieval-Augmented Generation

## Overview

Handles all vector storage: hierarchical scoping, static indexing, dynamic injection from tool results, embedding generation, and the Qdrant backend.

## Hierarchical Scope Model (ADR-018)

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

## Files

| File | Purpose |
|------|---------|
| `scopes.py` | `OrchidRAGScope` dataclass + `build_qdrant_filter()` |
| `dynamic.py` | `inject_to_rag()` — indexes MCP/built-in tool results |
| `indexer.py` | `StaticIndexer` — batch indexes test/seed data at startup |
| `embeddings.py` | `LiteLLMEmbedder` — generates embeddings via LiteLLM |
| `factory.py` | `build_reader()` — creates OrchidVectorReader from settings |
| `null.py` | `NullVectorReader` — no-op backend for tests / RAG-disabled mode |
| `backends/qdrant.py` | `QdrantRepository` — full Qdrant implementation |

## Key Rules

- **Never import `qdrant_client` outside `backends/qdrant.py`.** Other files use `OrchidVectorReader`/`OrchidVectorWriter` from `core/repository.py`.
- **Embedding model dimensions are critical.** `nomic-embed-text` = 768-d, `text-embedding-3-small` = 1536-d. Changing models requires re-creating Qdrant collections.
- **`inject_to_rag()` is safe to call on `NullVectorReader`** — it checks `isinstance(reader, OrchidVectorWriter)` first.
- **Namespaces = Qdrant collection names.** One collection per domain (e.g., `learning`, `notifications`, `uploads`). Tenant isolation is via payload filtering, NOT separate collections.

## QdrantRepository Notes

- Collections are created at startup via `ensure_collections()`.
- Uses `FieldCondition` with `MatchAny`/`MatchValue` for scope filtering.
- `duplicate_with_new_scope()` — copies points with new metadata (used by chat sharing).
- `scroll_by_scope()` — scrolls matching points (used to find chat-scoped data).
- Embedding happens lazily inside `upsert()` — documents without embeddings are embedded on the fly.
