# core/ — Pure Abstractions

## Golden Rule

**This package must have ZERO external dependencies.** Only Python stdlib imports are allowed. Every other package in the project depends on `core/`. Adding a third-party import here is an architectural bug (ADR-008).

Allowed: `dataclasses`, `typing`, `abc`, `enum`, `datetime`, `collections`, etc.
Forbidden: `qdrant_client`, `litellm`, `asyncpg`, `langchain`, `pydantic`, etc.

## Files

### `state.py` — AuthContext + AgentState

```python
AuthContext                 # Base class — subclass to add platform-specific fields.
  .access_token            # raw bearer token
  .expires_at              # epoch seconds (0 = no expiry)
  .extra                   # dict — extension point for arbitrary data
  # Framework contract (override in subclasses):
  .tenant_key → str        # tenant identifier for RAG scoping (default: "default")
  .user_id → str           # user identifier for chat ownership
  .is_expired → bool
  .bearer_header → dict    # {"Authorization": "Bearer {token}"}

# Consumer subclass example:
MyPlatformAuthContext(AuthContext)
  .domain                  # e.g. "acme.example.com"
  ._tenant_id              # Platform tenant ID
  .user_uuid               # Platform user UUID

AgentState(TypedDict)      # Flows through LangGraph. Extended by GraphState.
  .messages                # list of LangChain messages
  .auth_context            # AuthContext
  .chat_id                 # current chat session UUID
  .active_agents           # agents activated this round
  .mcp_context             # dict of tool results (agent_name → data)
  .rag_context             # dict of retrieved chunks
  .final_response          # set when supervisor is done
  .skill_instructions      # per-agent instructions for orchestrator skills
```

### `repository.py` — Vector Store Interfaces

```python
VectorReader(ABC)          # Read-only. Agents depend on this.
  .retrieve(query, namespace, k, scope) → list[SearchResult]

VectorWriter(ABC)          # Write-only. Indexers depend on this.
  .upsert(documents, namespace)
  .delete(document_ids, namespace)

VectorStoreRepository(VectorReader, VectorWriter)  # Combined. Qdrant implements this.
```

The `scope` parameter in `retrieve()` is a `RAGScope` — NOT a raw dict. If you see `filters: dict` anywhere, it's legacy and should be migrated.

### `agent.py` — BaseAgent ABC

```python
BaseAgent(ABC)
  .name → str              # unique identifier (e.g., "learning")
  .description → str       # Supervisor reads this to decide routing
  .rag_namespace → str     # Qdrant collection name
  .run(state) → AgentState # the main agent logic
  .extract_user_query(state) → str
  .fetch_rag_context(query, scope, namespace, k) → list[dict]
  .summarise(state, system_prompt, context_str, model) → str
```

### `mcp.py` — MCPClient ABC

```python
MCPClient(ABC)
  .call_tool(tool_name, arguments, auth) → dict
  .list_tools(auth) → list
```

## When Modifying core/

- Run `ruff check src/core/` to verify no forbidden imports crept in.
- Any new dataclass should be `frozen=True` if it represents identity/config.
- `TypedDict` with `total=False` for state objects (all fields optional for partial updates).
- Don't add default implementations to ABCs — keep them pure contracts.
