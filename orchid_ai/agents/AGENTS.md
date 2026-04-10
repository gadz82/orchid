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
summary = await self.summarise(query, mcp_data, rag_data, system_prompt=PROMPT)

# BAD — duplicated private methods
user_query = self._extract_user_query(state)  # don't copy-paste from BaseAgent
rag_data = await self._fetch_rag_context(...)  # use self.fetch_rag_context()
```

### Use `LLMProvider` for summarization

For simple LLM completions (summarization, classification), use the inherited `self.summarise()` which routes through the injected `LLMProvider`:

```python
# GOOD — uses LLMProvider via BaseAgent
summary = await self.summarise(query, mcp_data, rag_data, system_prompt=MY_PROMPT)

# BAD — direct litellm import for simple completion
import litellm
response = await litellm.acompletion(model=model, messages=[...])
```

### Agentic loops: litellm is acceptable

When you need tool_calls from the LLM response (multi-turn agentic loops), use `litellm` directly with a **lazy import inside the method**:

```python
async def _agentic_loop(self, ...):
    # Agentic loop requires full response objects (tool_calls),
    # so we use litellm directly — LLMProvider.complete() only returns str.
    import litellm
    from orchid_ai.llm_service import get_llm_kwargs
    
    response = await litellm.acompletion(model=model, messages=msgs, tools=tools, **get_llm_kwargs(model))
```

Never import `litellm` at the module level in consumer agents.

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
- **Importing `litellm` at module level for summarization.** Use `self.summarise()` instead.
