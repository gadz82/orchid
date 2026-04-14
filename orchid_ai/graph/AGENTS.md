# graph/ — LangGraph Wiring

## Overview

Assembles the LangGraph state machine: supervisor routing, graph state with annotations, and dynamic graph construction from YAML config.

## Files

| File | Purpose |
|------|---------|
| `state.py` | `GraphState` — extends `AgentState` with LangGraph annotations |
| `supervisor.py` | Supervisor node — routing logic, synthesis, sequential/parallel |
| `graph.py` | `build_graph()` — dynamically constructs graph from YAML config |

## GraphState

```python
class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # auto-merge + dedup
    auth_context: AuthContext
    chat_id: str
    active_agents: list[str]
    execution_mode: Literal["parallel", "sequential"]
    pending_agents: Annotated[list[str], replace_list]    # NOT merged, replaced
    mcp_context: Annotated[dict, merge_dicts]             # shallow merge
    rag_context: Annotated[dict, merge_dicts]
    skill_instructions: Annotated[dict, merge_dicts]
    final_response: str | None
```

**Annotations matter:**
- `add_messages` — LangGraph's built-in message merger (deduplicates by ID)
- `merge_dicts` — shallow merge so parallel agents' results don't overwrite each other
- `replace_list` — `pending_agents` is replaced wholesale (not appended)

## Supervisor Routing

The supervisor decides routing via LLM:

```json
{"execution": "parallel",   "agents": ["learning", "notifications"]}
{"execution": "sequential", "agents": ["learning", "notifications"]}
{"execution": "skill",      "skill": "remind_incomplete_users"}
```

**Sequential flow:** Only the first agent is activated. After it returns to the supervisor, the next agent in `pending_agents` is activated. This enables dependent workflows (e.g., Learning finds users → Notifications creates alert for those users).

**Skill flow:** Expands the skill into a sequential pipeline with per-step instructions stored in `skill_instructions`.

## Graph Construction

```python
def build_graph(config: AgentsConfig, runtime: OrchidRuntime) -> CompiledGraph:
```

`OrchidRuntime` holds the resolved dependencies (reader, LLM provider, MCP client factory). Integrators override only what they need:

```python
from orchid import OrchidRuntime, build_graph

runtime = OrchidRuntime(
    default_model="openai/gpt-4o",
    reader=my_reader,
    llm_service=MyCustomProvider(),
    mcp_client_factory=lambda cfg: MyMCPClient(cfg.url),
)
graph = build_graph(config=config, runtime=runtime)
```

Steps:
1. Resolves dependencies from `OrchidRuntime` (or builds one from kwargs)
2. Creates `StateGraph(GraphState)`
3. Adds `supervisor` node
4. For each agent in YAML config: instantiates agent via runtime's MCP factory, adds as `{name}_agent` node
5. Adds conditional edges from supervisor (routing function)
6. Adds return edges from each agent back to supervisor
7. Compiles and returns

## Adding a Node

New agents are added by YAML config — `build_graph()` handles the wiring automatically. You should NOT manually edit `graph.py` to add nodes unless changing the graph topology itself (e.g., adding a new routing strategy).

## Conversation History in the Supervisor

The supervisor uses `BaseAgent.extract_conversation_history()` to build clean multi-turn context for routing, synthesis, and sequential advance steps. This filters out internal `[Supervisor` messages and truncates per-message content.

### Configurable History Limits

Configured via `SupervisorConfig` in `agents.yaml`:

```yaml
supervisor:
  history_max_turns: 20    # max user-assistant pairs (default: 20)
  history_max_chars: 1000  # max chars per message before truncation (default: 1000)
```

### Sliding-Window Summarization (Context Compression)

When `history_summary_enabled: true`, history exceeding the recent window is compressed into a summary paragraph via an LLM call. This dramatically reduces token usage for long conversations while preserving key context.

```yaml
supervisor:
  history_summary_enabled: true
  history_summary_model: gemini/gemini-2.5-flash-lite  # cheap model for summaries
  history_summary_recent_turns: 10  # keep last 10 exchanges verbatim
```

The supervisor calls `BaseAgent.compress_conversation_history()` in both `_synthesise()` and `_advance_sequential()`. `GenericAgent` also compresses in `_step_summarise()` when the config is present. On LLM failure, the system falls back to using only the recent turns (no crash).

### `_filter_internal_messages()`

Helper that removes supervisor routing noise (`[Supervisor] Parallel dispatch:`, `[Supervisor → agent]` handoffs, etc.) from the current turn's messages before passing them to the LLM. Used in `_route()`, `_synthesise()`, and `_advance_sequential()`.

### Prior Tool Context

The supervisor's `_advance_sequential()` injects `mcp_context` from previous agent invocations into the history, so downstream agents in a sequential pipeline have access to earlier tool results.

## Common Mistakes

- **Using `list` annotation instead of `replace_list` for `pending_agents`.** Default list annotation appends, but sequential routing needs full replacement.
- **Not returning to supervisor.** Every agent node must have an edge back to `supervisor`. The supervisor decides when to end.
- **Mutating state in-place.** Always return a new dict from nodes. LangGraph handles merging via annotations.
- **Hardcoding history limits.** Use `SupervisorConfig.history_max_turns` and `history_max_chars` instead of magic numbers.
