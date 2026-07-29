# graph/ — LangGraph Wiring

## Overview

Assembles the LangGraph state machine: supervisor routing, graph state with annotations, and dynamic graph construction from YAML config.

## Files

| File | Purpose |
|------|---------|
| `state.py` | `GraphState` — extends `OrchidAgentState` with LangGraph annotations |
| `supervisor.py` | Supervisor node — routing logic, synthesis, sequential/parallel |
| `synthesizer.py` | Response synthesis — merges sub-agent results into a final response |
| `sequential_advancer.py` | Sequential pipeline advancer — activates next agent in chain |
| `_supervisor_helpers.py` | Shared helpers: message filtering, LLM completion, single-agent detection |
| `graph.py` | `build_graph()` — dynamically constructs graph from YAML config, wires memory strategies |

## GraphState

```python
class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # auto-merge + dedup
    # NOTE: auth is NOT a state channel.  It travels in the RunnableConfig
    # (config["configurable"]["auth_context"]); node callables read it via
    # auth_from_config(config) and agents via self._current_auth.  Keeping
    # auth out of state means it is never written to a checkpoint.
    chat_id: str
    active_agents: list[str]
    execution_mode: Literal["parallel", "sequential"]
    pending_agents: Annotated[list[str], replace_list]  # NOT merged, replaced
    mcp_context: Annotated[dict, merge_dicts]  # shallow merge
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
{"execution": "parallel",   "agents": ["knowledge-base", "messaging"]}
{"execution": "sequential", "agents": ["knowledge-base", "messaging"]}
{"execution": "skill",      "skill": "notify_matching_users"}
```

**Sequential flow:** Only the first agent is activated. After it returns to the supervisor, the next agent in `pending_agents` is activated. This enables dependent workflows (e.g., one agent finds a set of matching users, the next agent creates alerts for them).

**Skill flow:** Expands the skill into a sequential pipeline with per-step instructions stored in `skill_instructions`.

## Graph Construction

```python
def build_graph(config: OrchidAgentsConfig, runtime: OrchidRuntime) -> CompiledGraph:
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

The supervisor uses `OrchidAgent.extract_conversation_history()` to build clean multi-turn context for routing, synthesis, and sequential advance steps. This filters out internal `[Supervisor` messages and truncates per-message content. All filtering is consolidated through `MessageFilterPipeline` from `core/message_filter.py` — every call site uses the same pipeline presets.

### Configurable History Limits

Configured via `OrchidSupervisorConfig` in `agents.yaml`:

```yaml
supervisor:
  history_max_turns: 20    # max user-assistant pairs (default: 20)
  history_max_chars: 1000  # max chars per message before truncation (default: 1000)
```

### Conversation Memory (Running Summary + RAG)

The memory system replaces the stateless O(n²) re-summarization with stateful incremental compression. Configured via `supervisor.memory:`:

```yaml
supervisor:
  memory:
    strategy: "rag_augmented"           # none | running_summary | rag_augmented
    summary_recent_turns: 10           # keep last N exchanges verbatim
    summary_model: null                # null = supervisor model
    structured_output: true            # JSON entity extraction
    persist_summary: true              # store in chat storage
    # -- rag_augmented only --
    rag_k: 5                           # relevant turns to retrieve
    rag_similarity_threshold: 0.5
    store_turns: true
    truncation_strategy: "hard"        # hard | middle | llm | semantic
    truncation_max_chars: 1000
```

**Strategy phases:**
1. `running_summary` — incremental LLM-based summary extension (avoids re-compute)
2. `rag_augmented` — adds Qdrant-backed semantic retrieval of past turns

### Sliding-Window Summarization (Context Compression)

When `history_summary_enabled: true`, history exceeding the recent window is compressed into a summary paragraph via an LLM call. This dramatically reduces token usage for long conversations while preserving key context.

```yaml
supervisor:
  history_summary_enabled: true
  history_summary_model: gemini/gemini-2.5-flash-lite  # cheap model for summaries
  history_summary_recent_turns: 10  # keep last 10 exchanges verbatim
```

The supervisor calls `OrchidAgent.compress_conversation_history()` in both `_synthesise()` and `_advance_sequential()`. `GenericAgent` also compresses in `_step_summarise()` when the config is present. On LLM failure, the system falls back to using only the recent turns (no crash).

### Message Filtering Pipeline

All message filtering across the codebase is consolidated in `MessageFilterPipeline` (`core/message_filter.py`):
- `SUPERVISOR_PIPELINE` — preset for routing/synthesis: skips `[Supervisor]`, `[Conversation summary]`, tool types, excludes last user message
- `agent_pipeline(prefixes)` — factory for per-agent filtering with strip_prefix

Previous ad-hoc filtering in 6+ locations (`_filter_internal_messages`, manual iteration in synthesizer/advancer) has been replaced.

### Prior Tool Context

The supervisor's `_advance_sequential()` injects `mcp_context` from previous agent invocations into the history, so downstream agents in a sequential pipeline have access to earlier tool results.

## Mini-agent topology

When an agent has `mini_agent.enabled: true` in its YAML, the graph
builder synthesises **two extra nodes** alongside the normal
`{name}_agent`:

```
supervisor ──Send──> {name}_agent  ──conditional──>  supervisor                  (no fork)
                                                 └──> [Send×N] {name}_mini ──> {name}_aggregator ──> supervisor
```

Three additions in `graph.py`:

1. `_create_agent_node(agent, ..., agent_config=...)` runs the
   decomposer hook (`maybe_decompose()` from
   `agents/mini_agent_decomposer.py`) **before** `agent.run()`.  If
   the decomposer chose to fork, the wrapper returns the decision
   state update with no `AIMessage` and the conditional edge fans
   out — `agent.run()` is never invoked that turn.
2. `_make_fork_router(parent_name)` reads
   `state["mini_agent_decisions"][parent_name]` and returns either
   `"supervisor"` (no fork) or a list of `Send(f"{parent}_mini",
   payload)` — one per sub-task.  Each Send carries the per-mini
   sentinel keys (`_active_mini_parent`, `_active_mini_id`,
   `_active_mini_subtask`, `_active_mini_tool_subset`) so the mini
   node can identify its sub-task without threading kwargs through
   LangGraph.
3. The mini + aggregator node factories live in
   `agents/mini_agent_{node,aggregator}.py` — the graph builder
   imports their factories and wires them in.

**Zero overhead for non-opt-in agents** — the wrapper short-circuits
the decomposer call when `mini_agent.enabled` is false, and the
mini/aggregator nodes are not added to the graph.

**Works with any `OrchidAgent` subclass.**  The earlier `isinstance(
agent, GenericAgent)` guard was dropped: the only requirement is
that the agent expose `_chat_model` and `mcp_clients` (both
inherited from the base class).  Custom classes like the helpdesk
`SupportAgent` opt in via YAML alone.

State channels for the fan-out:

- `mini_agent_decisions: Annotated[dict, merge_dicts]` — per-parent
  `MiniAgentDecomposition.model_dump()`, keyed by parent name.
- `mini_agent_outcomes: Annotated[dict, merge_dicts]` — per-mini
  `MiniAgentOutcome.model_dump()`, keyed by `f"{parent}#{mini_id}"`.
  This is the **shadow-slot convention** — every parallel writer
  owns a unique key so the shallow merge is race-free.

Streaming surface — four `mini_agent.*` SSE events flow through a
piggyback `SystemMessage` (see `orchid_ai/observability/mini_agent_events.py`).
The `orchid-api` streaming router strips them out of the user-visible
synthesis stream and re-emits them as SSE frames.

## Common Mistakes

- **Using `list` annotation instead of `replace_list` for `pending_agents`.** Default list annotation appends, but sequential routing needs full replacement.
- **Not returning to supervisor.** Every agent node must have an edge back to `supervisor`. The supervisor decides when to end.
- **Mutating state in-place.** Always return a new dict from nodes. LangGraph handles merging via annotations.
- **Hardcoding history limits.** Use `OrchidSupervisorConfig.history_max_turns` and `history_max_chars` instead of magic numbers.
- **Forgetting that mini-agent `bind_tools` is filtered by `tool_subset`.** The mini node passes `tool_subset` to `AgenticLoop`, which filters `all_tool_defs` AND `tool_map` defensively before `bind_tools`.  If you reach into the mini's loop directly, respect the same filter.
