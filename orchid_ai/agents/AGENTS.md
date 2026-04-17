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
2. Subclass `BaseAgent` from `orchid_ai.core.agent`
3. Use **absolute imports** (not relative): `from orchid_ai.core.agent import BaseAgent`
4. Reference in `agents.yaml` with `class: examples.helpdesk.agents.support.SupportAgent`

Example custom agents:
- `examples/helpdesk/agents/support.py` — `SupportAgent` (agentic KB search loop)
- `examples/restaurant/agents/reviews.py` — `ReviewsAgent` (sentiment analysis + RAG)

## Agent Contract

- **Input:** `AgentState` (TypedDict)
- **Output:** Partial `AgentState` — must include `messages` with an `AIMessage`
- **Constructor:** `__init__(self, *, name, llm, reader, mcp_clients, config)` (injected by the framework)
- **Resolution:** Dotted import path in YAML `class:` field → `importlib` at startup

## SOLID Patterns for Agents

### Use inherited helpers — don't duplicate

`BaseAgent` provides reusable methods. Custom agents must use them instead of reimplementing:

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
user_query = self._extract_user_query(state)  # don't copy-paste from BaseAgent
rag_data = await self._fetch_rag_context(...)  # use self.fetch_rag_context()
```

### Use `extract_conversation_history()` for multi-turn context

`BaseAgent.extract_conversation_history(state)` is a static method that extracts clean user/assistant pairs from the graph state. It filters out `[Supervisor` routing messages, strips agent name prefixes, excludes the current query, and caps output to `max_turns` pairs. Use it in custom agents to maintain conversation continuity across turns.

### Pass `conversation_history` and `prior_tool_context` to `summarise()`

`GenericAgent` automatically passes these to `summarise()`. Custom agents should do the same:

- **`conversation_history`** — List of `{"role": ..., "content": ...}` dicts from `extract_conversation_history()`. Injected between system and user messages.
- **`prior_tool_context`** — Dict from `state["mcp_context"][agent_name]`. Appended to the system prompt so the agent remembers tool results from previous invocations.

### Use `BaseChatModel` for LLM calls

All LLM access goes through LangChain's `BaseChatModel` (injected as `chat_model=`):

```python
# GOOD — uses BaseChatModel via BaseAgent.summarise()
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

## Common Mistakes

- **Using relative imports in custom agents.** Files outside `src/` must use absolute imports (`from orchid_ai.core.agent import BaseAgent`).
- **Not returning `messages`.** The supervisor expects an `AIMessage` from each agent.
- **Forgetting `name=self.name` on `AIMessage`.** The supervisor uses this to track responses.
- **Duplicating `_extract_user_query()` or `_fetch_rag_context()`.** Use the inherited `self.extract_user_query()` and `self.fetch_rag_context()`.
- **Importing `litellm` directly.** Use `self._chat_model.ainvoke()` or `self.summarise()` instead.
