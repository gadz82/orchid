# agents/ — Agent Framework

## Overview

Contains the **library-provided** agent infrastructure. The library ships `GenericAgent` (config-driven, handles the standard 6-step flow from YAML) and the package marker.

**Concrete custom agents live outside `src/`** — in `examples/<project>/agents/` or in consumer projects. They are resolved at runtime via dotted import paths in `agents.yaml`.

## GenericAgent — The Standard Agent

`GenericAgent` in `generic_agent.py` handles the full 6-step flow from YAML config alone:

```
1. RAG retrieval      → fetch relevant context from Qdrant (scoped)
2. Skill check        → if query matches an agent skill → run skill → skip to 6
3. MCP tool calls     → call configured MCP servers (all/sequential/llm_decides)
4. Built-in tool calls→ run in-process Python functions
5. Dynamic RAG inject → write tool results back to Qdrant for future retrieval
6. LLM summarisation  → synthesize final response with all gathered context
```

Most agents should use `GenericAgent` via YAML — no Python needed.

## Custom Agents (outside src/)

When YAML isn't enough, create a custom agent:

1. Create a file in `examples/<project>/agents/` (or your own project)
2. Subclass `OrchidAgent` from `orchid_ai.core.agent`
3. Use **absolute imports** (not relative): `from orchid_ai.core.agent import OrchidAgent`
4. Reference in `agents.yaml` with `class: examples.helpdesk.agents.support.SupportAgent`

Example custom agents:
- `examples/helpdesk/agents/support.py` — `SupportAgent` (agentic KB search loop)
- `examples/restaurant/agents/reviews.py` — `ReviewsAgent` (sentiment analysis + RAG)

## Agent Contract

- **Input:** `OrchidAgentState` (TypedDict)
- **Output:** Partial `OrchidAgentState` — must include `messages` with an `AIMessage`
- **Constructor:** `__init__(self, *, reader, mcp_clients, config, chat_model=None, graph_store=None, **kwargs)` — the graph builder injects `reader`, MCP clients, the per-agent `BaseChatModel`, and the runtime's `OrchidGraphStore` (`None` or a `NullGraphStore` when no graph backend is wired). `OrchidAgent`'s `**_kwargs` catch-all absorbs framework-injected extras (e.g. `model_id`, `summary_config`).
- **Resolution:** Dotted import path in YAML `class:` field → `importlib` at startup

### Per-Tool RAG Override

`OrchidAgentConfig.effective_rag(tool_name) -> OrchidRAGConfig`
returns the merged RAG config that should govern a specific tool.
Resolution order:

1. MCP tools — `mcp_servers[*].tools[*]` matching `tool_name`.
2. Built-in tools — looked up in `agent.builtin_tool_configs`
   (cached at validation time from `OrchidAgentsConfig.tools`).
3. Falls back to `agent.rag` when no override is set.

When an override exists, the tool's `model_dump(exclude_unset=True)`
overlays onto the agent's full RAG dump via a deep merge — so an
override of just `ingestion: {chunk_size: 400}` keeps every other
nested field (strategy, overlap, retrieval, namespace, …) intact.

`GenericAgent._step_dynamic_injection` calls `effective_rag` per
injectable tool and threads the resolved namespace + ingestion
strategy into `inject_to_rag`. Custom agents that override
`run()` and write tool results to RAG should follow the same
pattern (see [`rag/AGENTS.md`](../rag/AGENTS.md#per-tool-rag-override)).

## SOLID Patterns for Agents

### Use inherited helpers — don't duplicate

`OrchidAgent` provides reusable methods. Custom agents must use them instead of reimplementing:

```python
# GOOD — use inherited methods
user_query = self.extract_user_query(state)
rag_data = await self.fetch_rag_context(query, scope)
history = self.extract_conversation_history(state, max_turns=10, max_chars=1000)
prior_ctx = (state.get("mcp_context") or {}).get(self.name)
summary = await self.summarise(
    query, mcp_data, rag_data,
    system_prompt=PROMPT,
    conversation_history=history or None,
    prior_tool_context=prior_ctx,
)

# BAD — duplicated private methods
user_query = self._extract_user_query(state)  # don't copy-paste from OrchidAgent
rag_data = await self._fetch_rag_context(...)  # use self.fetch_rag_context()
```

### Use `extract_conversation_history()` for multi-turn context

`OrchidAgent.extract_conversation_history(state)` is a static method that extracts clean user/assistant pairs from the graph state. It filters out `[Supervisor` routing messages, strips agent name prefixes, excludes the current query, and caps output to `max_turns` pairs. Use it in custom agents to maintain conversation continuity across turns.

### Pass `conversation_history` and `prior_tool_context` to `summarise()`

`GenericAgent` automatically passes these to `summarise()`. Custom agents should do the same:

- **`conversation_history`** — List of `{"role": ..., "content": ...}` dicts from `extract_conversation_history()`. Injected between system and user messages.
- **`prior_tool_context`** — Dict from `state["mcp_context"][agent_name]`. Appended to the system prompt so the agent remembers tool results from previous invocations.

### Use `BaseChatModel` for LLM calls

All LLM access goes through LangChain's `BaseChatModel` (injected as `chat_model=`):

```python
# GOOD — uses BaseChatModel via OrchidAgent.summarise()
summary = await self.summarise(query, mcp_data, rag_data, system_prompt=MY_PROMPT)

# GOOD — direct ainvoke for custom logic
result = await self._chat_model.ainvoke(messages)

# GOOD — agentic loop with tool-calling
model_with_tools = self._chat_model.bind_tools(tool_defs)
ai_msg = await model_with_tools.ainvoke(messages)
# ai_msg.tool_calls = [{"name": ..., "args": ..., "id": ...}]
```

Do NOT import `litellm` directly in consumer agents. Use the injected `_chat_model` for all LLM calls.

### GenericAgent internal structure (SRP)

`GenericAgent` delegates to single-responsibility collaborators:
- `SkillDetector` — matches user queries to agent skills
- `MCPDispatcher` — discovers MCP capabilities and calls tools
- `SkillExecutor` — executes skill steps and agent delegation

Don't merge these back into `GenericAgent`. If you need to modify skill detection, edit `skill_detector.py`.

### Mini-agents

Three sibling modules implement opt-in self-cloning fan-out, kept
deliberately separate from `GenericAgent` to honour SRP:

- `mini_agent_decomposer.py` — `MiniAgentSubTask`,
  `MiniAgentDecomposition`, the `MiniAgentDecomposer` class, and the
  free function `maybe_decompose()`.  Runs a deterministic
  structured-output LLM call that decides whether the request
  decomposes into independent sub-tasks; enforces the parent's
  `tool_allowlist_mode` against the decomposer's `allowed_tools`.
- `mini_agent_node.py` — `MiniAgentOutcome` model and
  `mini_agent_node_factory()`.  Runs ONE focused agentic loop per
  ``Send`` against the parent's MCP clients with a curated tool
  subset; wraps the loop in `asyncio.wait_for(timeout)`; converts
  exceptions to `status="failed"` outcomes; **re-raises
  `langgraph.errors.GraphBubbleUp`** (HITL `interrupt()`) so
  LangGraph's runtime can pause the graph for approval.
- `mini_agent_aggregator.py` — `aggregator_node_factory()`.
  Synthesises the parent's final `AIMessage` from the per-mini
  outcomes; short-circuits with a deterministic error message when
  `0 / N` outcomes succeeded; merges `tool_results` from successful
  outcomes only into `mcp_context[parent_name]`.

The decomposer hook lives at the **graph-wrapper** level
(`graph._create_agent_node`), invoked via the `maybe_decompose()`
free function.  This lets ANY `OrchidAgent` subclass — `GenericAgent`
or a custom class with its own `run()` — opt in via YAML
`mini_agent.enabled: true`, without coordinating with its own
`run()` implementation.  The mini node bypasses the parent's
`run()` entirely; it builds its own focused system prompt
(`parent.prompt + sub_task.instruction + tools list`) and runs an
`AgenticLoop` with `is_mini=True` and the resolved `tool_subset`.

Don't merge these back into `GenericAgent`. The graph-level wrapper
is the only integration point.

## Common Mistakes

- **Using relative imports in custom agents.** Files outside `src/` must use absolute imports (`from orchid_ai.core.agent import OrchidAgent`).
- **Not returning `messages`.** The supervisor expects an `AIMessage` from each agent.
- **Forgetting `name=self.name` on `AIMessage`.** The supervisor uses this to track responses.
- **Duplicating `_extract_user_query()` or `_fetch_rag_context()`.** Use the inherited `self.extract_user_query()` and `self.fetch_rag_context()`.
- **Importing `litellm` directly.** Use `self._chat_model.ainvoke()` or `self.summarise()` instead.
