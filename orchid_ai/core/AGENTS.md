# core/ — Pure Abstractions

## Golden Rule

**This package depends only on `langchain-core`** (for `Document` and message types). No concrete backend imports are allowed.

Allowed: `dataclasses`, `typing`, `abc`, `enum`, `datetime`, `collections`, `langchain_core.*`
Forbidden: `qdrant_client`, `litellm`, `asyncpg`, `pydantic` (except via langchain_core)

## Files

### `state.py` — OrchidAuthContext + OrchidAgentState

```python
OrchidAuthContext                 # Base class — subclass to add platform-specific fields.
  .access_token            # raw bearer token
  .expires_at              # epoch seconds (0 = no expiry)
  .extra                   # dict — extension point for arbitrary data
  # Framework contract (override in subclasses):
  .tenant_key → str        # tenant identifier for RAG scoping (default: "default")
  .user_id → str           # user identifier for chat ownership
  .is_expired → bool
  .bearer_header → dict    # {"Authorization": "Bearer {token}"}

# Consumer subclass example:
MyPlatformAuthContext(OrchidAuthContext)
  .domain                  # e.g. "acme.example.com"
  ._tenant_id              # Platform tenant ID
  .user_uuid               # Platform user UUID

OrchidAgentState(TypedDict)      # Flows through LangGraph. Extended by GraphState.
  .messages                # list of LangChain messages
  .auth_context            # OrchidAuthContext
  .chat_id                 # current chat session UUID
  .active_agents           # agents activated this round
  .mcp_context             # dict of tool results (agent_name → data)
  .rag_context             # dict of retrieved chunks
  .final_response          # set when supervisor is done
  .skill_instructions      # per-agent instructions for orchestrator skills
```

### `repository.py` — Vector Store Interfaces

```python
OrchidVectorReader(ABC)          # Read-only. Agents depend on this.
  .retrieve(query, namespace, k, scope) → list[OrchidSearchResult]

OrchidVectorWriter(ABC)          # Write-only. Indexers depend on this.
  .upsert(documents, namespace)
  .delete(document_ids, namespace)

OrchidVectorStoreRepository(OrchidVectorReader, OrchidVectorWriter)  # Combined. Qdrant implements this.
```

The `scope` parameter in `retrieve()` is a `OrchidRAGScope` — NOT a raw dict.  Raw `filters: dict` is not supported.

### `memory.py` — OrchidConversationMemory ABC

```python
OrchidConversationMemory(ABC)     # Conversation memory strategies.
  .get_running_summary(chat_id) → str | None
  .update_running_summary(chat_id, new_messages, existing_summary) → str
  .get_relevant_history(query, chat_id, k) → list[dict]
  .store_conversation_turn(chat_id, tenant_id, user_id, turn, metadata) → None

NullConversationMemory           # No-op when memory is disabled (strategy: "none").
```

### `memory_types.py` — Structured Summary Models

```python
OrchidConversationSummary       # Dataclass: topics, entities, actions, decisions, questions, preferences, narrative
  .to_context_string() → str    # Render for LLM injection
  .to_dict() / .from_dict()     # JSON round-trip
  .from_json(json_str)          # Parse JSON → OrchidConversationSummary
  .merge(existing, new_data)    # Merge with entity deduplication

OrchidSummaryEntity             # Dataclass: name, type, details
```

### `message_filter.py` — Unified Filtering Pipeline

```python
MessageFilter                   # Single filter: skip_prefix | strip_prefix | max_chars | skip_type | exclude_last_user
MessageFilterPipeline           # Applies a sequence of filters to LangGraph messages → clean dicts
  .apply(messages, truncation_strategy) → list[dict]

SUPERVISOR_PIPELINE             # Preset: skips [Supervisor], [Conversation summary], tool types, excludes last user
agent_pipeline(prefixes, ...)   # Factory for per-agent pipelines with strip_prefix
```

All message filtering across the codebase (6+ call sites) now delegates to this single pipeline. See `extract_conversation_history()`.

### `truncation.py` — Message Truncation Strategies

```python
OrchidTruncationStrategy(str, Enum)  # hard | middle | llm | semantic
truncate_content(content, max_chars, strategy) → str           # synchronous
truncate_content_async(content, max_chars, strategy, ...) → str  # async (LLM/SEMANTIC)
```

- `hard` — `content[:max_chars] + "…"` (current behavior)
- `middle` — keeps first 40% and last 40%, `…[truncated]…` in between
- `llm` — asks LLM to summarize; falls back to `middle` on failure
- `semantic` — reserved for embedding-based selection; falls back to `middle`

### `agent.py` — OrchidAgent ABC

```python
OrchidAgent(ABC)
  .name → str              # unique identifier (e.g., "knowledge-base")
  .description → str       # Supervisor reads this to decide routing
  .rag_namespace → str     # Qdrant collection name
  .run(state) → OrchidAgentState # the main agent logic
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
- **Truncates** individual messages using the configured `truncation_strategy` (default `"hard"` = `content[:max_chars] + "…"`; also `"middle"` which preserves start+end, `"llm"` for LLM summarization)
- Accepts an optional `pipeline: MessageFilterPipeline` parameter for custom filter chains
- Uses **duck-typing** for message type detection to maintain zero-dependency rule in `core/`

#### `compress_conversation_history()` — Sliding-Window Summarization

Async static method that implements the **sliding-window with summarization** pattern. When conversation history exceeds `recent_turns * 2` messages, older turns are compressed into a single LLM-generated summary paragraph, while the most recent `recent_turns` exchanges are preserved verbatim.

```python
compressed = await OrchidAgent.compress_conversation_history(
    history,
    chat_model=llm_provider,
    recent_turns=10,                        # keep last 10 exchanges verbatim
    running_summary=existing_summary,       # extend incrementally (Phase 1)
    structured_output=True,                 # JSON with entities (Phase 2)
    compression_system_prompt="...",        # custom system prompt (Phase 4)
    extension_user_prompt="...",            # custom extension prompt (Phase 4)
)
# Result: [{"role": "assistant", "content": "[Conversation summary]\n..."}, ...recent messages]
```

- If history fits within the window, returns it **unchanged** (no LLM call)
- On LLM failure, **falls back** to returning only the recent turns (no summary, no crash)
- Uses `temperature=0.0` for deterministic summaries
- When `running_summary` is provided, uses an **extension prompt** (incremental update) instead of summarizing from scratch — avoids O(n²) token waste
- When `structured_output=True`, produces JSON with entity extraction; falls back to narrative on parse failure
- All prompts overridable via `OrchidAgentPromptConfig` in `agents.yaml` (`summary_compression_*` fields)

#### `summarise()` — Conversation-Aware Summarization

When `conversation_history` (list of dicts) is provided, `summarise()` injects the history between the system prompt and user message, and appends a focus instruction telling the LLM to prioritize the latest query.

When `prior_tool_context` (dict) is provided, it appends `--- Previous Tool Results ---` JSON to the system prompt (truncated to 4000 chars). This carries tool results from previous invocations so agents don't re-ask for information already gathered.

### `mcp.py` — OrchidMCPClient ABC

```python
OrchidMCPClient(ABC)
  .call_tool(tool_name, arguments, auth) → dict
  .list_tools(auth) → list
```

## When Modifying core/

- Run `ruff check src/core/` to verify no forbidden imports crept in.
- Any new dataclass should be `frozen=True` if it represents identity/config.
- `TypedDict` with `total=False` for state objects (all fields optional for partial updates).
- Don't add default implementations to ABCs — keep them pure contracts.
