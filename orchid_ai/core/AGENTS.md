# core/ — Pure Abstractions

## Golden Rule

**This package depends only on `langchain-core`** (for `Document` and message types). No concrete backend imports are allowed.

Allowed: `dataclasses`, `typing`, `abc`, `enum`, `datetime`, `collections`, `langchain_core.*`
Forbidden: `qdrant_client`, `litellm`, `asyncpg`, `pydantic` (except via langchain_core)

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
  .summarise(query, mcp_data, rag_data, *, system_prompt, model, temperature,
             conversation_history, prior_tool_context) → str
  @staticmethod
  .extract_conversation_history(state, *, max_turns, max_chars,
                                skip_prefixes, strip_prefixes) → list[dict]
```

#### `extract_conversation_history()` — Framework-level History Extraction

Static method that extracts clean `[{"role": "user"|"assistant", "content": "..."}]` pairs from the LangGraph state's `messages` list. Used by `GenericAgent`, the supervisor, and custom agents to build multi-turn context.

- **Filters out** internal supervisor messages (any `AIMessage` starting with `skip_prefixes`, default `("[Supervisor",)`)
- **Strips agent prefixes** from AI messages (e.g. `"[MyAgent]\nActual content"` → `"Actual content"`)
- **Excludes** the last `HumanMessage` (it's the current query, added separately)
- **Caps** output to `max_turns * 2` messages (default `max_turns=10`)
- **Truncates** individual messages to `max_chars` characters with `…` suffix (default `None` = no truncation)
- Uses **duck-typing** for message type detection to maintain zero-dependency rule in `core/`

#### `compress_conversation_history()` — Sliding-Window Summarization

Async static method that implements the **sliding-window with summarization** pattern. When conversation history exceeds `recent_turns * 2` messages, older turns are compressed into a single LLM-generated summary paragraph, while the most recent `recent_turns` exchanges are preserved verbatim.

```python
compressed = await BaseAgent.compress_conversation_history(
    history,
    llm_service=llm_provider,
    model="gemini/gemini-2.5-flash-lite",  # use a cheap/fast model
    recent_turns=10,                        # keep last 10 exchanges verbatim
)
# Result: [{"role": "assistant", "content": "[Conversation summary]\n..."}, ...recent messages]
```

- If history fits within the window, returns it **unchanged** (no LLM call)
- On LLM failure, **falls back** to returning only the recent turns (no summary, no crash)
- Uses `temperature=0.0` for deterministic summaries
- The summary prompt asks the LLM to focus on: key topics, entities, actions taken, outstanding questions

#### `summarise()` — Conversation-Aware Summarization

When `conversation_history` (list of dicts) is provided, `summarise()` injects the history between the system prompt and user message, and appends a focus instruction telling the LLM to prioritize the latest query.

When `prior_tool_context` (dict) is provided, it appends `--- Previous Tool Results ---` JSON to the system prompt (truncated to 4000 chars). This carries tool results from previous invocations so agents don't re-ask for information already gathered.

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
