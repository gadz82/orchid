<p align="center">
  <img src="icon.svg" alt="Orchid" width="80" />
</p>

<h1 align="center">Orchid</h1>

A platform-agnostic multi-agent AI framework built on [LangGraph](https://github.com/langchain-ai/langgraph) and [LiteLLM](https://github.com/BerriAI/litellm).

Orchid (alias for Orchestrator-Index) lets you define AI agents via YAML configuration, orchestrate them with a supervisor, connect external tools via MCP servers, and augment responses with hierarchical RAG — all without writing agent code.

## BETA - This is a work in progress.

## Features

- **YAML-driven agents** — define agents, tools, skills, and prompts in `agents.yaml`
- **Multi-provider LLM** — OpenAI, Anthropic, Google Gemini, Groq, Ollama via LiteLLM
- **Hierarchical RAG** — 5-level scoping (shared, tenant, user, chat, agent) with Qdrant built-in support
- **Pluggable retrieval strategies** — `simple`, `multi_query`, `hyde`, `hybrid`, `graph_rag` plus integrator-registered custom strategies
- **Pluggable query transformers** — `reformulate`, `multi_query`, `hyde`, `decompose`, all with configurable system prompts
- **MCP tool integration** — connect to external services via Streamable HTTP MCP servers, with `none` / `passthrough` / `oauth` auth modes (OAuth covers RFC 9728 / RFC 8414 / RFC 7591 DCR)
- **Built-in tools** — register Python functions as in-process tools, with declarative parameter metadata
- **Agent skills** — multi-step workflows within agents and across agents (orchestrator skills)
- **Mini-agents (self-clone fork)** — opt-in per-agent decomposer + aggregator that fan a single turn into independent sub-tasks running in parallel
- **Parallel tool dispatch** — opt-in intra-round parallel tool calls based on per-tool `parallel_safe` annotations
- **Per-tool RAG caching** — opt-in `inject_to_rag` with configurable TTL per tool
- **Internal prompt customisation** — every supervisor / synthesis / agent / RAG-transformer / mini-agent / summarise prompt is YAML- and Python-configurable with backwards-compatible defaults
- **Sliding-window history summarisation** — opt-in compression of older turns by a cheaper LLM so long conversations stay within budget
- **AI Guardrails** — 3-tier safety layer (global input, per-agent, global output) with built-in prompt injection, PII, content safety, topic restriction, max length, and groundedness checks
- **Pluggable persistence** — SQLite (default) and PostgreSQL backends for chat history; integrators can plug any `OrchidChatStorage` subclass
- **HITL graph interrupts** — `requires_approval: true` tools pause the graph; resume via the API or CLI with the user's decision
- **MCP capability cache warming** — `OrchidSessionWarmer` keeps tool inventories ready so the first agentic round avoids discovery RPCs
- **Pollen + Bloom (event-driven activation)** — opt-in async substrate that turns external webhooks, cron schedules, and in-graph `emit_signal` calls into background LangGraph runs. Triggers match signals to agent invocations under a synthesised `OrchidAuthContext`; run results can be appended back into a real user chat.
- **Document pipeline** — PDF, DOCX, XLSX, CSV, image parsing with pluggable ingestion strategies and post-processors

## Installation

```bash
pip install orchid-ai
```

With PostgreSQL support:

```bash
pip install "orchid-ai[postgres]"
```

## Quick Start

### 1. Define Agents

Create an `agents.yaml`:

```yaml
version: "1"

defaults:
  llm:
    model: ollama/llama3.2
    temperature: 0.2

agents:
  assistant:
    description: "General-purpose assistant"
    prompt: |
      You are a helpful AI assistant.
      Answer questions clearly and concisely.
```

### 2. Use Programmatically

```python
from orchid_ai import OrchidRuntime, build_graph, load_config

config = load_config("agents.yaml")
runtime = OrchidRuntime(default_model="ollama/llama3.2")
graph = build_graph(config=config, runtime=runtime)

result = await graph.ainvoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "auth_context": auth,
})
```

### 3. Or Use via orchid-cli / orchid-api

This library is consumed by:

- **[orchid-api](https://github.com/gadz82/orchid-api)** -- FastAPI HTTP server
- **[orchid-cli](https://github.com/gadz82/orchid-cli)** -- Typer command-line interface
- **[orchid-frontend](https://github.com/gadz82/orchid-frontend)** -- Next.js chat UI

## Architecture

```
orchid/
  core/             Pure ABCs -- ZERO external dependencies (only stdlib)
    agent.py        OrchidAgent ABC
    state.py        OrchidAuthContext + OrchidAgentState
    identity.py     OrchidIdentityResolver ABC
    llm_provider.py LLMProvider ABC
    mcp.py          OrchidMCPToolCaller / OrchidMCPDiscoverable ABCs
    repository.py   OrchidVectorReader / OrchidVectorWriter / OrchidVectorStoreAdmin ABCs
  config/           YAML config loader + Pydantic schema + registries
  agents/           GenericAgent + collaborators (SkillDetector, MCPDispatcher, SkillExecutor)
  graph/            LangGraph supervisor + graph builder
  rag/              Scoping, indexing, embeddings, dynamic injection, Qdrant backend
  documents/        PDF/DOCX/XLSX/CSV/Image parsers + chunking pipeline
  persistence/      OrchidChatStorage ABC + SQLite (default) + PostgreSQL backends + migrations
  mcp/              StreamableHttpMCPClient
  events/           Pollen + Bloom — concrete impls of core/events ABCs:
                      backends/, queues/, processors/, runners/, producers/,
                      schedulers/, auth/, registry, dispatcher, streaming
  identity/         OAuthMintingMixin (helper for resolvers used by act_as_user triggers)
  llm_service.py    LiteLLMProvider (concrete LLMProvider)
  utils.py          Shared utilities
```

### Dependency Direction

```
graph/ -> agents/ -> core/
          agents/ -> rag/ -> core/
          agents/ -> mcp/ -> core/
persistence/ -> core/
documents/   -> core/
```

`core/` is the leaf -- it has ZERO external dependencies. Only Python stdlib imports.

## Core ABCs

| ABC | File | Purpose |
|-----|------|---------|
| `OrchidAgent` | `core/agent.py` | Agent identity, `run()`, `summarise()`, `fetch_rag_context()`, `extract_conversation_history()` |
| `OrchidIdentityResolver` | `core/identity.py` | Bearer token → `OrchidAuthContext` (per-request validation **and** the `/auth/resolve-identity` bridge) |
| `OrchidAuthConfigProvider` | `core/auth_config.py` | Resolves non-secret upstream-OAuth discovery |
| `OrchidAuthExchangeClient` | `core/auth_config.py` | Server-side authorization-code + refresh-token exchange |
| `OrchidMCPToolCaller` | `core/mcp.py` | Call MCP tools |
| `OrchidMCPDiscoverable` | `core/mcp.py` | Discover MCP capabilities |
| `OrchidMCPTokenStore` | `core/mcp.py` | Per-user outbound OAuth token persistence |
| `OrchidMCPClientRegistrationStore` | `core/mcp.py` | Per-server discovered endpoints + DCR creds |
| `OrchidMCPGatewayClientStore` / `…AuthCodeStore` / `…TokenStore` | `core/mcp_gateway_state.py` | Inbound MCP gateway state (DCR clients, in-flight auth codes, issued tokens) |
| `OrchidVectorReader` | `core/repository.py` | Vector store retrieval |
| `OrchidVectorWriter` | `core/repository.py` | Vector store indexing |
| `OrchidVectorStoreAdmin` | `core/repository.py` | Collection management |
| `OrchidChatStorage` | `persistence/base.py` | Chat CRUD + message persistence |
| `OrchidSignalDispatcher` | `core/events/dispatcher.py` | Persist + enqueue a `SignalEnvelope` (Pollen ingest) |
| `OrchidSignalQueue` | `core/events/queue.py` | Durable signal buffer (in-memory / SQLite / Postgres / relay) |
| `OrchidSignalProducer` | `core/events/producer.py` | Surface external events as signals (HTTP / scheduler / internal) |
| `OrchidSignalProcessor` | `core/events/processor.py` | Drain the queue, match triggers, execute Blooms |
| `OrchidJobRunner` | `core/events/runner.py` | Invoke the LangGraph supervisor under a synthesised auth context |
| `OrchidSignalStore` / `OrchidJobStore` / `OrchidScheduleStore` / `OrchidTriggerStore` | `core/events/store.py` | Per-table stores backing the events tables |

The auth ABCs (`OrchidAuthConfigProvider`, `OrchidAuthExchangeClient`,
`OrchidIdentityResolver`, three `OrchidMCPGateway*Store`s) collectively let
`orchid-api` host every secret-bearing OAuth call on behalf of downstream
public PKCE clients (the MCP gateway, Next.js frontends).

## OrchidRuntime

`OrchidRuntime` is the single integration point for consumers. It holds all resolved
dependencies needed by `build_graph()` — override only what you need, everything else
gets a sensible default.

```python
from orchid_ai import OrchidRuntime, build_graph, load_config
```

### Minimal (all defaults)

Uses `LiteLLMProvider` for LLM, `NullVectorReader` (no RAG), and `StreamableHttpMCPClient`
for MCP servers:

```python
config = load_config("agents.yaml")
runtime = OrchidRuntime(default_model="ollama/llama3.2")
graph = build_graph(config=config, runtime=runtime)
```

### Custom Vector Store

Plug in a Qdrant-backed reader (or any `OrchidVectorReader` implementation):

```python
from orchid_ai.rag.factory import build_reader

reader = build_reader(vector_backend="qdrant", qdrant_url="http://localhost:6333")

runtime = OrchidRuntime(
    default_model="gemini/gemini-2.5-flash",
    reader=reader,
)
```

### Custom LLM Provider

Replace the default `LiteLLMProvider` with your own `LLMProvider` implementation:

```python
from orchid_ai.core.llm_provider import LLMProvider

class MyProvider(LLMProvider):
    async def complete(self, *, model: str, messages: list, temperature: float = 0.2) -> str:
        # your custom logic
        ...

runtime = OrchidRuntime(
    default_model="my-model",
    llm_service=MyProvider(),
)
```

### Custom MCP Client Factory

Control how MCP clients are created from server config entries:

```python
runtime = OrchidRuntime(
    default_model="ollama/llama3.2",
    mcp_client_factory=lambda cfg: MyMCPClient(cfg.url, api_key=MY_KEY),
)
```

### All Options

```python
runtime = OrchidRuntime(
    default_model="openai/gpt-4o",        # LiteLLM model identifier
    reader=my_qdrant_reader,               # OrchidVectorReader | None
    llm_service=MyCustomProvider(),        # LLMProvider | None
    mcp_client_factory=my_factory,         # Callable[[OrchidMCPServerConfig], OrchidMCPClient] | None
)
graph = build_graph(config=config, runtime=runtime)
```

| Field | Type | Default |
|-------|------|---------|
| `default_model` | `str` | `"ollama/llama3.2"` |
| `reader` | `OrchidVectorReader \| None` | `NullVectorReader` (no RAG) |
| `llm_service` | `LLMProvider \| None` | `LiteLLMProvider()` |
| `mcp_client_factory` | `MCPClientFactory \| None` | `StreamableHttpMCPClient` factory |

## Configuration

Orchid uses two configuration files:

- **`agents.yaml`** -- Agent definitions, tools, skills, and supervisor (managed by the library)
- **`orchid.yml`** -- Runtime settings for LLM, RAG, storage, auth, and tracing (managed by orchid-api/orchid-cli)

**Priority:** env vars > `orchid.yml` > hardcoded defaults.

---

### agents.yaml Reference

#### Root Level

| Field | Type | Default |
|-------|------|---------|
| `version` | str | `"1"` |
| `defaults` | object | |
| `tools` | dict | `{}` |
| `skills` | dict | `{}` |
| `supervisor` | object | |
| `guardrails` | object | `{}` |
| `agents` | dict | (required) |

- **`version`** -- Schema version string. Currently always `"1"`. Reserved for future backward-compatible migrations.
- **`defaults`** -- Default LLM and RAG settings inherited by every agent. Agents can override any default individually. Avoids repeating the same model or RAG config across all agents.
- **`tools`** -- Global registry of built-in Python tools. Each tool is a named entry mapping to a Python function. Agents reference tools by name in their `tools` list. Tools declared here are available to any agent that includes their name.
- **`skills`** -- Orchestrator-level (cross-agent) multi-step workflows. The supervisor detects when a user query matches a skill and runs agents in sequence, passing results forward. Useful for complex tasks that span multiple domains (e.g. "plan a trip" involving flights + hotels + activities).
- **`supervisor`** -- Customization of the supervisor node that routes queries to agents, synthesizes multi-agent responses, and manages orchestrator skills. Override prompts here to change routing logic without modifying code.
- **`guardrails`** -- Global input and output guardrail chains. Input guardrails run on every user message before the supervisor; output guardrails run on every response before returning to the user. See "Guardrails" section below.
- **`agents`** -- The core of the config: a dictionary of agent definitions keyed by name. Each agent is a self-contained unit with its own prompt, tools, MCP connections, RAG settings, guardrails, and skills. At least one agent is required.

#### `defaults.llm`

| Field | Type | Default |
|-------|------|---------|
| `model` | str | `"gemini/gemini-2.5-flash"` |
| `temperature` | float | `0.2` |
| `fallback_model` | str\|null | `null` |

- **`model`** -- The LLM model identifier using LiteLLM's `provider/model-name` format. This is the default model used by all agents unless overridden per-agent. Supported providers include `ollama/llama3.2` (local Ollama), `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `gemini/gemini-2.5-flash`, `groq/llama-3.3-70b-versatile`, and any model supported by LiteLLM.
- **`temperature`** -- Controls randomness in LLM responses. `0.0` = fully deterministic (always picks the most likely token), `1.0` = maximum randomness. Lower values (0.1--0.3) are best for factual/tool-calling agents. Higher values (0.7--0.9) suit creative tasks. Default `0.2` favors consistency.
- **`fallback_model`** -- Optional fallback LLM model. When set, the framework automatically retries with this model if the primary model fails (503, rate limit, timeout). Disabled by default (`null`). When set at `defaults.llm` level, it applies to all agents and the supervisor unless overridden per-agent or per-supervisor. Example: `"ollama/llama3.2"` as fallback for a cloud model.

#### `defaults.rag`

| Field | Type | Default |
|-------|------|---------|
| `k` | int | `5` |
| `enabled` | bool | `true` |
| `rag_ttl` | int | `0` |

- **`k`** -- Maximum number of documents retrieved from the vector store per agent query. When an agent runs, it embeds the user's query and performs a cosine similarity search in Qdrant. `k` controls how many of the top-scoring chunks are returned and injected into the LLM prompt as context. Higher values provide more context but consume more tokens and risk including irrelevant results. A flight search agent might use `k: 10` for broad coverage, while a FAQ agent might use `k: 3` for precision.
- **`enabled`** -- Master switch for RAG retrieval across all agents. When `false`, no agent queries the vector store and no dynamic injection occurs. Useful for demos or agents that rely entirely on tools. Individual agents can override this.
- **`rag_ttl`** -- Default time-to-live (in seconds) for tool results cached in RAG. When a tool has `inject_to_rag: true`, its results are stored in Qdrant with a timestamp. On subsequent queries, if cached results exist that are newer than `rag_ttl` seconds ago, the framework reuses them instead of re-calling the tool. `0` = caching disabled (tools are always called fresh). Individual tools can override this value.

#### `supervisor`

| Field | Type | Default |
|-------|------|---------|
| `assistant_name` | str | `"AI assistant"` |
| `fallback_model` | str\|null | `null` |
| `routing_system_prompt` | str | `null` |
| `synthesis_system_prompt` | str | `null` |
| `sequential_advance_prompt` | str | `null` |
| `history_max_turns` | int | `20` |
| `history_max_chars` | int | `1000` |
| `history_summary_enabled` | bool | `true` |
| `history_summary_model` | str | `null` |
| `history_summary_recent_turns` | int | `10` |

- **`assistant_name`** -- The name used in the supervisor's prompts when referring to itself (e.g. "You are the routing brain of **Travel Assistant**"). Appears in synthesized responses. Set this to your product's name.
- **`fallback_model`** -- Optional fallback LLM for the supervisor specifically. Overrides `defaults.llm.fallback_model` for routing, synthesis, and sequential advance calls. Use when the supervisor needs a different fallback than the agents (e.g. the supervisor uses a fast model with a reliable fallback, while agents use a powerful model with a cheaper fallback).
- **`routing_system_prompt`** -- Fully custom system prompt for the supervisor's routing step. The routing step analyzes the user's message and decides which agent(s) should handle it by reading each agent's `description`. When `null`, the built-in template from `supervisor.py` is used. Override this to change how agents are selected (e.g. to add domain-specific routing rules or prioritization logic).
- **`synthesis_system_prompt`** -- Custom system prompt for the synthesis step. After all selected agents return their results, the supervisor synthesizes them into a single coherent response. Override this to control the tone, format, or structure of final responses.
- **`sequential_advance_prompt`** -- Custom prompt used during orchestrator skill execution. After each step in a multi-agent skill completes, this prompt decides whether to advance to the next step or respond directly. Override this to change how skill steps chain together.
- **`history_max_turns`** -- Maximum number of user-assistant conversation pairs included as context in supervisor routing, synthesis, and sequential advance steps. Each "turn" is one user message + one assistant response. Higher values give more context but consume more tokens. Default `20`.
- **`history_max_chars`** -- Maximum characters per individual message in conversation history. Messages exceeding this limit are truncated with an ellipsis (`…`). Prevents long tool outputs or verbose responses from consuming excessive tokens in multi-turn context. Default `1000`.
- **`history_summary_enabled`** -- Enables sliding-window conversation summarization. When `true`, conversation turns older than `history_summary_recent_turns` are compressed into a single LLM-generated summary paragraph, while the most recent turns are kept verbatim. This dramatically reduces token usage for long conversations. Default `true`. Set to `false` to disable.
- **`history_summary_model`** -- LLM model used for the history summarization call. Use a cheap/fast model here since the summarization input is small. When `null`, the supervisor's default model is used. Example: `"gemini/gemini-2.5-flash-lite"`.
- **`history_summary_recent_turns`** -- Number of recent user-assistant exchange pairs to keep verbatim when summarization is enabled. Older turns are condensed into a summary. Default `10` (the last 10 exchanges are preserved word-for-word, everything older is summarized).

#### `tools.<name>` (Built-in Tools)

| Field | Type | Default |
|-------|------|---------|
| `handler` | str | (required) |
| `description` | str | `""` |
| `parameters` | dict | `{}` |
| `inject_to_rag` | bool | `false` |
| `rag_ttl` | int\|null | `null` |

- **`handler`** -- Dotted Python import path to the tool function (e.g. `"myapp.tools.weather.get_weather"`). The function is imported via `importlib` at graph build time. It must be callable with keyword arguments `query` and `context`, and must be importable from the working directory.
- **`description`** -- Human-readable description of what the tool does. This is included in the LLM prompt so the model understands when and how to use the tool. A good description helps the LLM decide whether to call this tool for a given query. Be specific: "Get current weather temperature and conditions for a city name" is better than "Weather tool".
- **`parameters`** -- Optional parameter declarations for the tool. When provided, these take precedence over auto-extracted parameters from the function signature. Each parameter is a dict with `type` (string/int/float/bool), `description`, `required` (bool), and `default`. When omitted, parameters are auto-extracted from the Python function signature via `inspect` — framework-injected params (`query`, `context`, `auth_context`, `**kwargs`) are filtered out automatically. This metadata is used by the CLI skill generator (`orchid skill generate`) to produce accurate Claude Code skill documentation.
- **`inject_to_rag`** -- When `true`, the tool's return value is stored as a document in the Qdrant vector store after execution. This creates a cache: on future queries, the framework can retrieve the cached result from RAG instead of re-calling the tool (if `rag_ttl > 0`). Useful for expensive API calls whose results don't change frequently (e.g. catalog snapshots, reference data). Default `false` means results are used once and discarded.
- **`rag_ttl`** -- Per-tool override for the RAG cache time-to-live (in seconds). When `null`, the agent's `rag.rag_ttl` is used. When set to a positive integer, this tool's cached results expire after that many seconds. Set to `0` to disable caching for this specific tool even if the agent has a default TTL. Useful when different tools have different freshness requirements (e.g. exchange rates: 300s, restaurant menus: 86400s).

#### `skills.<name>` (Orchestrator Skills)

| Field | Type | Default |
|-------|------|---------|
| `description` | str | `""` |
| `steps` | list | (required) |

- **`description`** -- Human-readable description of the entire workflow. The supervisor's LLM reads this to decide whether to activate the skill for a given user query. Write it as a summary of the end-to-end outcome: "Plan a complete trip: find flights, book hotels, and suggest activities at the destination."
- **`steps`** -- Ordered list of agent invocations. Each step runs one agent, and the results are passed to the next step as context. Steps execute sequentially -- the output of step 1 is available to step 2's agent.

Each step:

| Field | Type |
|-------|------|
| `agent` | str |
| `instruction` | str |

- **`agent`** -- Name of the agent to invoke (must match a key in the `agents` dict).
- **`instruction`** -- Specific instruction or question passed to the agent for this step. This overrides the user's original query for this step. For example: "Based on the flight results, find hotels near the airport for those dates." The agent receives both this instruction and the accumulated results from previous steps.

#### `agents.<name>`

| Field | Type | Default |
|-------|------|---------|
| `description` | str | (required) |
| `prompt` | str | (required) |
| `class` | str | `null` |
| `llm` | object | (from defaults) |
| `rag` | object | (from defaults) |
| `tools` | list[str] | `[]` |
| `mcp_servers` | list | `[]` |
| `skills` | dict | `{}` |
| `guardrails` | object | `{}` |
| `execution_hints` | object | |
| `children` | dict | `null` |

- **`description`** -- Short description of the agent's domain and capabilities. The supervisor reads this to decide which agent(s) should handle a user's query. Write it from the supervisor's perspective: "Flight search and booking agent. Searches airlines, compares prices, and can hold reservations." A vague description leads to poor routing; a precise one ensures the right agent is selected.
- **`prompt`** -- The system prompt sent to the LLM when this agent runs. Defines the agent's personality, expertise, and behavior rules. This is the most important field for controlling agent output quality. Include what the agent should focus on, how it should use tool results, and what format to use for responses.
- **`class`** -- Dotted Python import path to a custom `OrchidAgent` subclass (e.g. `"myapp.agents.hotels.HotelAgent"`). When `null` (the default), the built-in `GenericAgent` is used, which handles the standard 6-step flow (RAG retrieval, skill check, MCP tools, built-in tools, dynamic injection, LLM summarization) entirely from YAML config. Only set this when you need custom Python logic that `GenericAgent` can't express (e.g. agentic loops, custom API integrations, complex state management).
- **`llm`** -- Per-agent LLM override with `model` and `temperature`. When set, this agent uses a different model than the default. Useful for assigning cheaper/faster models to simple agents and more capable models to complex ones. When `null`, inherits from `defaults.llm`.
- **`rag`** -- Per-agent RAG settings (see `agents.<name>.rag` below). Each agent can have its own vector store namespace, retrieval depth, and cache TTL.
- **`tools`** -- List of built-in tool names (strings) available to this agent. These reference tools declared in the root `tools` section. The agent's `GenericAgent` will call each listed tool during step 4 of its pipeline and include the results in the LLM context.
- **`mcp_servers`** -- List of MCP server connections (see `agents.<name>.mcp_servers[]` below). Each server provides external tools, prompts, and resources via the Model Context Protocol.
- **`skills`** -- Agent-level multi-step workflows (see `agents.<name>.skills.<name>` below). Unlike orchestrator skills (which span multiple agents), these are internal to one agent and chain tool calls or sub-agent invocations within the agent's domain.
- **`guardrails`** -- Per-agent input and output guardrail chains. These run in addition to global guardrails when this specific agent is active. Use for domain-specific enforcement like topic restrictions. See "Guardrails" section below.
- **`execution_hints`** -- Hints that the supervisor uses when routing. Currently only `parallel_safe` (see below).
- **`children`** -- Recursive sub-agent definitions. Allows nesting agents under a parent. Sub-agents inherit the parent's defaults and are included in the supervisor's routing. Useful for organizing related agents hierarchically.

#### `agents.<name>.rag`

| Field | Type | Default |
|-------|------|---------|
| `namespace` | str | `""` |
| `k` | int | `5` |
| `enabled` | bool | `true` |
| `rag_ttl` | int | `0` |

- **`namespace`** -- The Qdrant collection name where this agent's domain knowledge is stored (e.g. `"flights"`, `"hotels"`, `"knowledge_base"`). Each namespace is a separate Qdrant collection. Multiple agents can share a namespace (they'll see the same data within their scope), or each can have its own. Leave empty (`""`) if the agent doesn't use RAG retrieval. The namespace is also used for dynamic injection -- tool results with `inject_to_rag: true` are stored in this collection.
- **`k`** -- Maximum number of documents retrieved from Qdrant per query for this agent. Overrides `defaults.rag.k`. The agent embeds the user's query, performs cosine similarity search in its namespace, and returns the top `k` results. These are injected into the LLM prompt as context. Higher values give more context but cost more tokens and may dilute relevance. Tune per agent based on corpus size and query type.
- **`enabled`** -- Whether this agent queries the vector store. When `false`, steps 1 (RAG retrieval) and 5 (dynamic injection) are skipped entirely. The agent relies only on tools and its prompt. Override `defaults.rag.enabled` for agents that don't need vector search (e.g. a simple calculator agent).
- **`rag_ttl`** -- Cache TTL (seconds) for tool results injected into RAG by this agent. Overrides `defaults.rag.rag_ttl`. When a tool with `inject_to_rag: true` runs, results are stored with a timestamp. On future queries, if cached results newer than `rag_ttl` seconds exist, the tool is skipped and cached data is used instead. `0` = always call tools fresh. Individual tools can further override this with their own `rag_ttl`.

#### `agents.<name>.execution_hints`

| Field | Type | Default |
|-------|------|---------|
| `parallel_safe` | bool | `true` |

- **`parallel_safe`** -- Tells the supervisor whether this agent can run concurrently with other agents. When `true` (default), the supervisor may invoke multiple agents in parallel for a single query (e.g. asking both a flights agent and a hotels agent simultaneously). When `false`, the supervisor runs this agent sequentially. Set to `false` when the agent depends on results from other agents, has side effects, or when tool execution order matters.

#### `agents.<name>.mcp_servers[]`

| Field | Type | Default |
|-------|------|---------|
| `name` | str | (required) |
| `type` | `"local"` / `"remote"` | `"local"` |
| `transport` | `"streamable_http"` / `"sse"` | `"streamable_http"` |
| `url` | str | (required) |
| `auth` | object | `{mode: "none"}` |
| `tools` | list / `"*"` | `[]` |
| `prompts` | list / `"*"` | `[]` |
| `resources` | list / `"*"` | `[]` |
| `tool_call_strategy` | `"all"` / `"sequential"` / `"llm_decides"` | `"all"` |

- **`name`** -- Unique identifier for this MCP server within the agent. Used in logging, error messages, and as a key when referencing the server in skill steps (`source: "airline-api"`). Must be unique per agent.
- **`type`** -- Whether the MCP server runs as a local process (`"local"`) or as a remote HTTP service (`"remote"`). Local servers are co-deployed with the agent (e.g. in the same Docker network). Remote servers are external services accessed over the network. This affects connection handling and error retry behavior.
- **`transport`** -- The MCP transport protocol. `"streamable_http"` is the standard stateless protocol (recommended). `"sse"` uses Server-Sent Events for streaming responses. Most MCP servers use `streamable_http`.
- **`url`** -- The MCP server's HTTP endpoint. Supports environment variable interpolation with `${VAR_NAME}` syntax (e.g. `"${AIRLINE_MCP_URL}"`). Variables are resolved from the environment at config load time.
- **`tools`** -- Either an explicit list of `ToolConfig` objects (specifying which tools to use from this server) or the wildcard `"*"` to auto-discover all tools at runtime via `list_tools()`. An explicit list acts as an allow-list: only listed tools are called, even if the server offers more. Use `"*"` for development/exploration; use explicit lists in production for predictability and security.
- **`prompts`** -- Prompt template names to load from the MCP server, or `"*"` to load all. Prompts are predefined query templates that the server provides (e.g. a `"catalog_schema"` prompt that returns the data schema). Loaded prompts are included in the agent's context.
- **`resources`** -- Resource URIs to load from the MCP server, or `"*"` to load all. Resources are static data endpoints (e.g. `"catalog/"` returns a list of available items). Loaded resources are included in the agent's context.
- **`tool_call_strategy`** -- Controls how multiple tools on this server are executed:
  - `"all"` -- Call every tool in the list simultaneously and collect all results. Fastest, but tools run independently without seeing each other's output.
  - `"sequential"` -- Call tools one by one in order. Each tool receives the accumulated results from previous tools as a `previous_results` argument. Use when tools depend on each other (e.g. search then filter then sort).
  - `"llm_decides"` -- Ask the LLM to decide which tools to call and with what arguments. The LLM sees all available tools and the user query, then generates tool calls. Most flexible but slower and uses more tokens.
- **`auth`** -- Per-server authentication configuration (see `agents.<name>.mcp_servers[].auth` below). Determines how the client authenticates with this MCP server. Defaults to `mode: "none"` (no auth headers).

> **Capability cache lifetime:** discovery results (`list_tools()`, `list_prompts()`, `list_resources()`) are cached for the lifetime of the process and warmed proactively at startup / session start by `OrchidSessionWarmer` -- the per-request hot path stops paying the discovery cost. Flush stale capabilities via `OrchidMCPClient.invalidate_cache()` (or a future admin endpoint).

> **Fault isolation:** MCP server communication boundaries use broad exception handling. If a server returns HTTP errors (401 Unauthorized, 500 Internal Server Error), connection failures, or protocol errors, the agent logs a warning and continues with the remaining servers and tools -- it does not crash or retry endlessly. This applies to tool execution (strategies), capability discovery (`render_capabilities`), and the `fetch()` dispatcher. One failing MCP server never takes down the entire agent.

#### `agents.<name>.mcp_servers[].tools[]`

| Field | Type | Default |
|-------|------|---------|
| `name` | str | (required) |
| `arguments` | dict | `{}` |
| `inject_to_rag` | bool | `false` |
| `rag_ttl` | int\|null | `null` |

- **`name`** -- The exact tool name as registered on the MCP server. Must match what the server reports via `list_tools()`. This is the identifier used when calling `client.call_tool(name, args, auth)`.
- **`arguments`** -- Default arguments passed to this tool on every invocation. These are merged with the query and any strategy-specific arguments. Useful for tools that always need a fixed parameter (e.g. `currency: USD`, `language: en`, `max_results: 10`). The agent can't override these at runtime -- they're baked into the config.
- **`inject_to_rag`** -- When `true`, the tool's return value is stored as a document in Qdrant after execution. This enables the RAG cache: on subsequent queries within the same chat scope, the framework checks if cached results exist before re-calling the tool. Default `false` -- results are used once for the LLM response and then discarded. Enable for tools whose results are expensive to compute and don't change frequently.
- **`rag_ttl`** -- Per-tool override for the cache TTL (seconds). When `null`, uses the agent's `rag.rag_ttl`. When set to a positive integer, cached results from this tool expire after that many seconds. Set to `0` to disable caching for this tool even if the agent has a default TTL. Useful when tools have different freshness requirements within the same agent.

#### `agents.<name>.mcp_servers[].auth` (MCP Auth)

| Field | Type | Default |
|-------|------|---------|
| `mode` | `"none"` / `"passthrough"` / `"oauth"` | `"none"` |

YAML carries ONLY the auth mode. Nothing else — no `client_id`, no
`client_secret`, no endpoints — needs to live in configuration.

- **`mode`** -- How the MCP client authenticates with this server:
  - `"none"` (default) -- No authentication headers. Use for local MCP servers or remote servers without auth.
  - `"passthrough"` -- Forwards the graph's `OrchidAuthContext` bearer token unchanged. Use when the MCP server trusts the same identity provider as the main application.
  - `"oauth"` -- Per-user OAuth 2.0 flow with the MCP server's authorization server. The framework follows the **MCP 2025-03-26 authorization spec**: on the first 401 it consumes the `WWW-Authenticate: Bearer resource_metadata="…"` header (RFC 9728), fetches the authorization server metadata (RFC 8414), dynamically registers a client (RFC 7591), and persists the resulting endpoints + credentials to `OrchidMCPClientRegistrationStore`. Per-user tokens land in `OrchidMCPTokenStore` and are refreshed against the discovered token endpoint automatically.

The authorization server MUST advertise `registration_endpoint` in its
RFC 8414 metadata. If it doesn't, discovery fails with a clear error —
integrators whose IdP lacks DCR should seed `OrchidMCPClientRegistrationStore`
manually with the relevant endpoints + client credentials before first use.

Example:

```yaml
mcp_servers:
  # No auth (default) -- local MCP server
  - name: local-tools
    url: http://localhost:3001/mcp
    tools: "*"

  # Passthrough -- forwards the platform bearer token
  - name: internal-api
    url: ${INTERNAL_MCP_URL}
    tools: "*"
    auth:
      mode: passthrough

  # OAuth -- everything discovered at runtime from the MCP server's 401
  - name: external-crm
    url: ${CRM_MCP_URL}
    tools: "*"
    auth:
      mode: oauth
```

#### `agents.<name>.skills.<name>` (Agent Skills)

| Field | Type | Default |
|-------|------|---------|
| `description` | str | `""` |
| `steps` | list | (required) |

- **`description`** -- Description of what this skill does. The agent's `SkillDetector` uses an LLM to match the user's query against available skill descriptions. If a match is found, the skill runs instead of the normal tool-calling pipeline. Write descriptions that clearly state the workflow: "Search the menu with a dietary filter, then show today's specials that also match."
- **`steps`** -- Ordered list of steps. Each step is either a tool call or an agent invocation (exactly one of `tool` or `agent` must be set). Steps execute sequentially, and each step receives the accumulated results from all previous steps.

Each step:

| Field | Type |
|-------|------|
| `tool` | str |
| `source` | str |
| `arguments` | dict |
| `agent` | str |
| `instruction` | str |

- **`tool`** -- Name of the tool to call (MCP tool name or built-in tool name). Mutually exclusive with `agent`.
- **`source`** -- Where to find the tool. Set to an MCP server `name` (e.g. `"airline-api"`) for MCP tools, or `"builtin"` for built-in Python tools. When `null` or omitted, defaults to `"builtin"`.
- **`arguments`** -- Extra arguments passed to the tool for this specific step. Merged with the tool's default arguments from the server config. Useful for step-specific overrides (e.g. `max_results: 5` in a comparison step).
- **`agent`** -- Name of another agent to invoke directly (bypasses the supervisor). The invoked agent runs its full pipeline (RAG + tools + LLM) and its result chains forward to the next step. Mutually exclusive with `tool`.
- **`instruction`** -- Query or instruction sent to the invoked agent. Overrides the user's original message for this step. Use it to provide step-specific context: "Based on the player's stats and situation, assess their motivation and suggest mental strategies."

#### `events` (Pollen + Bloom — optional, opt-in)

Top-level block that wires the event-driven activation layer. **Omit it (or set `events.enabled: false`) and nothing in `orchid_ai/events/` runs** — no producers / processors are started, no DB rows are written, zero overhead.

| Field | Type | Default |
|-------|------|---------|
| `enabled` | bool | `false` |
| `store` | component ref (`{class, ...}`) | `null` (required when `enabled: true`) |
| `queue` | component ref + queue knobs | `null` (required when `enabled: true`) |
| `scheduler` | component ref | `null` |
| `producers` | list[component ref] | `[]` |
| `processors` | list[component ref + processor knobs] | `[]` (≥1 required when `enabled: true`) |
| `middleware` | list[component ref] | `[]` |
| `ingestion` | object — webhook source registry | `{sources: []}` |
| `schedules` | list — cron / interval entries | `[]` |
| `triggers` | list — signal → agent rules | `[]` |

- **`enabled`** -- Master switch. The full block is still parsed when `false` so typos in your YAML still fail loudly, but no runtime objects are constructed. Default `false` is the zero-overhead opt-out.
- **`store`** -- Backend for the seven events tables (`signals`, `signal_queue`, `signal_queue_dead_letter`, `triggers`, `schedules`, `job_runs`, `signal_sources`). Built-in choices: `orchid_ai.events.backends.sqlite.SQLiteEventStorage`, `orchid_ai.events.backends.postgres.PostgresEventStorage`. The migrations live alongside chat/MCP migrations in `persistence/migrations/v001_initial_schema.py` — one root migration covers all three concerns.
- **`queue`** -- Durable signal buffer. Built-ins: `orchid_ai.events.queues.inmemory.InMemorySignalQueue` (tests/demos), `orchid_ai.events.queues.sqlite.SQLiteSignalQueue` (single-process durable), `orchid_ai.events.queues.postgres.PostgresSignalQueue` (FOR UPDATE SKIP LOCKED, optional `pg_notify` on commit), `orchid_ai.events.queues.relay.RelayingSignalQueue` (publish-then-mark adapter for external buses). Tunable knobs: `notify_enabled` (default `true`), `poll_interval_ms` (default `200`), `lease_seconds` (default `30`), `max_attempts` (default `5`), `dead_letter_table` (default `signal_queue_dead_letter`).
- **`scheduler`** -- Cron / interval driver. Built-in: `orchid_ai.events.schedulers.apscheduler.APSchedulerBackend` (wraps `apscheduler.AsyncIOScheduler`, no SQLAlchemy — durability lives in the `schedules` table; APScheduler's in-memory jobstore is re-populated on every boot).
- **`producers`** -- Sources of signals. Built-ins: `orchid_ai.events.producers.http.HTTPIngestionProducer` (lazy-imports FastAPI; mounts at `mount: /signals`), `orchid_ai.events.producers.scheduler.SchedulerProducer` (drives the configured `scheduler`), `orchid_ai.events.producers.internal.InternalEmissionProducer` (wires `OrchidAgent.emit_signal` and `DispatcherSignalEmitter`), `orchid_ai.events.producers.relay_recovery.RelayRecoveryProducer` (periodic re-publish sweep when using `RelayingSignalQueue`).
- **`processors`** -- Drain the queue and run the matched Blooms. Built-in: `orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor`. Tunable knobs: `concurrency` (default `4`), `poll_interval_ms` (default `200`), `lease_seconds` (default `30`), `max_attempts` (default `5`), `drain_timeout_seconds` (default `10.0`).
- **`middleware`** -- Optional `SignalIngestMiddleware` chain that runs on every `dispatcher.ingest` call before persistence (e.g. enrichment, tagging). Each entry is a component ref.
- **`ingestion.sources`** -- Webhook source registry consumed by `HTTPIngestionProducer`. Each source has `id`, `validator: {class, secret_ref, extra_args}`, `allowed_types` (allow-list of signal types this source can emit). Built-in validators: `orchid_ai.events.auth.HMACValidator` (constant-time SHA-256 against the raw body so payloads can be parsed safely AFTER the signature check), `orchid_ai.events.auth.BearerValidator`. `secret_ref` accepts `env:VAR_NAME` to read from the environment.
- **`schedules[]`** -- Cron / interval rows persisted in the `schedules` table. Each: `id`, exactly one of `cron: "0 7 * * 1-5"` or `interval_seconds: 3600`, `trigger_id` (must point at a trigger with `on.signal == "cron"`), `identity` (discriminated union — see below), `enabled` (default `true`). The `SchedulerProducer` fires synthetic `cron` signals through `dispatcher.ingest` with `dedupe_key = "<schedule_id>:<fire_iso>"`.
- **`triggers[]`** -- Signal → agent rules. Each: `id`, `on: {signal, cron?, when?}`, `emits: {agent, prompt_template, identity, respect_chat_binding?, visibility?}`, `retry: {max, backoff, jitter, initial_delay_seconds, max_delay_seconds}`, `parallelism: per_user | per_tenant | unbounded` (default `per_user`).

##### `events.triggers[].on`

| Field | Type | Notes |
|-------|------|-------|
| `signal` | str | e.g. `"support.ticket.created"`, `"cron"`. Used for first-pass match. |
| `cron` | str \| null | Required when `signal == "cron"`, rejected otherwise. |
| `when` | str \| null | Optional **JMESPath** boolean expression evaluated against the `SignalEnvelope`. The expression is compiled at registration time — invalid JMESPath fails boot, not run-time. |

##### `events.triggers[].emits`

| Field | Type | Notes |
|-------|------|-------|
| `agent` | str | Agent name to invoke. Must exist in `agents:` — validated at registration. |
| `prompt_template` | str | Mustache-style `{{var}}` template rendered against the signal envelope (`{{tenant_key}}`, `{{payload.foo}}`, etc.). |
| `identity` | object (discriminated by `mode`) | See below — produces the `OrchidAuthContext` the Bloom runs under. |
| `respect_chat_binding` | bool (default `false`) | When `true` AND the signal carries a `ChatBinding` AND the resolved auth has write permission on the target chat, the run's final `AIMessage` lands in that chat with `metadata.origin="bloom"`. Rejected at validation when combined with `identity.mode: service_account` (no user-of-record). |
| `visibility` | `actor` \| `addressed` \| `tenant` \| `admin` \| `null` | Run / signal visibility level for the §26 visibility filter. `null` = inferred from identity (`act_as_user → actor`, `addressed_to_user → addressed`, `service_account → admin`). The (identity, visibility) compatibility matrix is enforced at config-load AND registration-time. |

##### `events.{schedules,triggers}.identity` — discriminated union

| `mode` | Extra fields | Behaviour |
|--------|-------------|-----------|
| `service_account` | `name: str` | The processor calls `OrchidIdentityResolver.resolve_service_account(name)`. The platform acts under a named service identity (e.g. a `digest-bot`). No user-of-record — incompatible with `respect_chat_binding: true`. |
| `addressed_to_user` | `service_account: str`, `user_id_from: str` (JMESPath) | Same service identity, but the resulting auth context is *tagged* with a `user_id` extracted from the signal payload. Used for user-scoped RAG / chat binding without impersonation. |
| `act_as_user` | `user_id_from: str` (JMESPath) | Full user impersonation. The processor calls `OrchidIdentityResolver.mint_for_user(tenant_key, user_id)`. The resolver is **probed at boot** — a resolver that can't mint at all (raises `MintingProbeUnsupportedError`) gets a deterministic boot-time failure naming both the trigger and the resolver class. |

##### `events.triggers[].retry`

| Field | Type | Default |
|-------|------|---------|
| `max` | int | `0` |
| `backoff` | `fixed` \| `linear` \| `exponential` | `exponential` |
| `jitter` | bool | `true` |
| `initial_delay_seconds` | float | `1.0` |
| `max_delay_seconds` | float | `300.0` |

Per-trigger retry of the supervisor invocation (distinct from queue retry which is governed by `events.queue.max_attempts`). Retries become **new `JobRun` rows** with `attempt_number + 1` — never in-place updates. The `(trigger_id, signal_id, attempt_number)` UNIQUE constraint is what gives Bloom its replay safety.

##### `events.triggers[].parallelism`

`per_user` (default), `per_tenant`, or `unbounded`. The asyncio worker pool serialises Blooms by this key to avoid races on shared per-user state (notably the MCP capability cache).

##### Cross-field validation

When `events.enabled: true`:

- `events.store` and `events.queue` are required.
- `events.processors` must have at least one entry.
- Every `schedule.trigger_id` must reference a trigger declared in this same file (forward references aren't supported).
- Every schedule's matching trigger must declare `on.signal: cron`.
- Every Pydantic model under `schema_events` uses `extra: forbid` — typos in keys surface as clear errors instead of silent drift.

##### Worked example

```yaml
events:
  enabled: true

  store:
    class: orchid_ai.events.backends.postgres.PostgresEventStorage
  queue:
    class: orchid_ai.events.queues.postgres.PostgresSignalQueue
    notify_enabled: true
    lease_seconds: 60
  scheduler:
    class: orchid_ai.events.schedulers.apscheduler.APSchedulerBackend

  producers:
    - class: orchid_ai.events.producers.http.HTTPIngestionProducer
      extra_args:
        mount: /signals
    - class: orchid_ai.events.producers.scheduler.SchedulerProducer
    - class: orchid_ai.events.producers.internal.InternalEmissionProducer

  processors:
    - class: orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor
      concurrency: 8
      lease_seconds: 60

  ingestion:
    sources:
      - id: support-system
        validator:
          class: orchid_ai.events.auth.HMACValidator
          secret_ref: env:SUPPORT_HMAC_SECRET
        allowed_types: [support.ticket.created, support.ticket.updated]

  schedules:
    - id: morning-digest-cron
      cron: "0 7 * * 1-5"
      trigger_id: morning-digest
      identity:
        mode: service_account
        name: digest-bot

  triggers:
    # Cron-driven Bloom — a digest assembled by a service identity
    - id: morning-digest
      on:
        signal: cron
        cron: "0 7 * * 1-5"
      emits:
        agent: notifications
        prompt_template: "Build the morning digest for {{tenant_key}}"
        identity:
          mode: service_account
          name: digest-bot
      retry: { max: 3, backoff: exponential, jitter: true }
      parallelism: unbounded

    # Webhook-driven Bloom that posts back into the originating user's chat
    - id: support-ticket-triage
      on:
        signal: support.ticket.created
        when: "payload.priority == 'high'"
      emits:
        agent: support
        prompt_template: |
          A new high-priority ticket arrived: {{payload.subject}}.
          Draft an initial reply.
        identity:
          mode: addressed_to_user
          service_account: support-bot
          user_id_from: payload.requester.id
        respect_chat_binding: true
      retry: { max: 5, backoff: exponential }
      parallelism: per_user
```

---

### orchid.yml Reference

Runtime configuration consumed by orchid-api and orchid-cli. Each nested YAML key maps to a flat environment variable. **Priority:** env vars > orchid.yml > hardcoded defaults.

#### `agents`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `agents.config_path` | `AGENTS_CONFIG_PATH` | `"agents.yaml"` |

- **`agents.config_path`** -- Path to the `agents.yaml` file (relative to working directory or absolute). This is the only required pointer between the two config files. orchid-api and orchid-cli read this to find agent definitions.

#### `llm`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `llm.model` | `LITELLM_MODEL` | `"ollama/llama3.2"` |
| `llm.ollama_api_base` | `OLLAMA_API_BASE` | |
| `llm.groq_api_key` | `GROQ_API_KEY` | `""` |
| `llm.gemini_api_key` | `GEMINI_API_KEY` | `""` |
| `llm.anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` |
| `llm.openai_api_key` | `OPENAI_API_KEY` | `""` |

- **`llm.model`** -- Default LLM model for the API/CLI runtime. This is used by the graph builder as the fallback model when an agent doesn't specify one in `agents.yaml`. Uses LiteLLM format: `provider/model-name`.
- **`llm.ollama_api_base`** -- Base URL for the Ollama server when using `ollama/*` models. Defaults to `http://localhost:11434` if not set. In Docker, typically `http://host.docker.internal:11434` to reach the host's Ollama instance.
- **`llm.groq_api_key`** -- API key for Groq cloud inference. Required when using `groq/*` models (e.g. `groq/llama-3.3-70b-versatile`).
- **`llm.gemini_api_key`** -- API key for Google Gemini models. Required when using `gemini/*` models. Also used for Gemini embedding models in the RAG section.
- **`llm.anthropic_api_key`** -- API key for Anthropic Claude models. Required when using `anthropic/*` models.
- **`llm.openai_api_key`** -- API key for OpenAI models. Required when using `openai/*` models. Also used for OpenAI embedding models (`text-embedding-3-small`) in the RAG section.

#### `auth`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `auth.dev_bypass` | `DEV_AUTH_BYPASS` | `false` |
| `auth.identity_resolver_class` | `IDENTITY_RESOLVER_CLASS` | `""` |
| `auth.domain` | `AUTH_DOMAIN` | `""` |

- **`auth.dev_bypass`** -- When `true`, the API skips Bearer token validation and uses a dummy `OrchidAuthContext` with tenant `"99999"` and user `"dev-user-00000000"`. All requests are allowed without authentication. **Never enable in production.** Useful for local development and testing without an OAuth provider.
- **`auth.identity_resolver_class`** -- Dotted import path to a custom `OrchidIdentityResolver` subclass (e.g. `"myapp.identity.MyIdentityResolver"`). The resolver receives the Bearer token from the `Authorization` header and returns an `OrchidAuthContext` with tenant/user information. When empty, only `dev_auth_bypass` works -- all other requests get a 503.
- **`auth.domain`** -- Default platform domain passed to the identity resolver when the `x-auth-domain` header is missing from the request. Used by resolvers that need to know which tenant instance to authenticate against. When empty, the resolver must get the domain from another source.

> **CLI OAuth support:** `orchid-cli` extends the `auth` section with an `auth.cli` subsection for OAuth 2.0 Authorization Code + PKCE login. This is a CLI-only feature -- the API uses its own FastAPI dependency injection for auth. See the [orchid-cli README](https://github.com/gadz82/orchid-cli#authentication) for details.

#### `startup`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `startup.hook` | `STARTUP_HOOK` | `""` |

- **`startup.hook`** -- Dotted import path to an async function called once during server startup, after the graph is built and storage is initialized. The function receives `reader` and `settings` as keyword arguments. Use it for one-time setup tasks like seeding the vector store, pre-loading data, or registering webhooks. Example: `"myapp.startup.seed_data"`.

#### `rag`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `rag.vector_backend` | `VECTOR_BACKEND` | `"qdrant"` |
| `rag.qdrant_url` | `QDRANT_URL` | `"http://qdrant:6333"` |
| `rag.embedding_model` | `EMBEDDING_MODEL` | `"text-embedding-3-small"` |
| `rag.openai_api_key` | `OPENAI_API_KEY` | `""` |
| `rag.gemini_api_key` | `GEMINI_API_KEY` | `""` |

- **`rag.vector_backend`** -- Which vector store backend to use. `"qdrant"` connects to a Qdrant server for full vector search and storage. `"null"` uses a no-op backend that returns empty results -- useful for demos, testing, or agents that don't need RAG. Future options may include `"aoss"` (Amazon OpenSearch Serverless).
- **`rag.qdrant_url`** -- HTTP URL of the Qdrant server (e.g. `"http://localhost:6333"` for local, `"http://qdrant:6333"` for Docker). Collections are auto-created at startup for all namespaces declared in `agents.yaml`.
- **`rag.embedding_model`** -- The model used to convert text into vectors for storage and retrieval. Must match the dimensionality of existing Qdrant collections. **Switching models requires wiping and re-indexing all collections** because different models produce different-sized vectors. Common options: `text-embedding-3-small` (1536-d, OpenAI), `ollama/nomic-embed-text` (768-d, local), `gemini/gemini-embedding-001` (3072-d, Google).
- **`rag.openai_api_key`** -- API key for OpenAI embedding models. Required when `embedding_model` is an OpenAI model (e.g. `text-embedding-3-small`). Can be the same key as `llm.openai_api_key`.
- **`rag.gemini_api_key`** -- API key for Gemini embedding models. Required when `embedding_model` is a Gemini model.

#### `upload`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `upload.vision_model` | `VISION_MODEL` | `""` |
| `upload.namespace` | `UPLOAD_NAMESPACE` | `"uploads"` |
| `upload.max_size_mb` | `UPLOAD_MAX_SIZE_MB` | `20` |
| `upload.chunk_size` | `CHUNK_SIZE` | `1000` |
| `upload.chunk_overlap` | `CHUNK_OVERLAP` | `200` |

- **`upload.vision_model`** -- LLM model used to extract text from images and scanned documents via visual understanding. When empty, the primary `llm.model` is used as fallback. Set this to a vision-capable model (e.g. `"ollama/minicpm-v"`, `"openai/gpt-4o"`) for better image/PDF parsing quality. Only used during document upload -- not for regular chat.
- **`upload.namespace`** -- Qdrant collection name where uploaded document chunks are stored. Defaults to `"uploads"`. All agents can access uploaded documents via the `"uploads"` namespace in their RAG retrieval (in addition to their own domain namespace). This provides a shared document space within a chat session.
- **`upload.max_size_mb`** -- Maximum allowed file upload size in megabytes. Requests with files larger than this are rejected with a 413 error. Default 20 MB is suitable for most documents. Increase for large PDFs or XLSX files.
- **`upload.chunk_size`** -- Target size (in tokens) for each text chunk when splitting uploaded documents. Documents are parsed into text, then split into overlapping chunks of this size for embedding and storage. Smaller chunks (500) give more precise retrieval but less context per result. Larger chunks (2000) give more context but may include irrelevant content. Default 1000 is a good balance.
- **`upload.chunk_overlap`** -- Number of overlapping tokens between consecutive chunks. Overlap ensures that concepts spanning a chunk boundary aren't lost. Default 200 means each chunk shares 200 tokens with its neighbor. Set to 0 for no overlap (faster indexing, potentially missed context at boundaries).

#### `storage`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `storage.class` | `CHAT_STORAGE_CLASS` | `"orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage"` |
| `storage.dsn` | `CHAT_DB_DSN` | `"~/.orchid/chats.db"` |

- **`storage.class`** -- Dotted import path to the `OrchidChatStorage` implementation. The class is dynamically imported at startup. Built-in options:
  - `orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage` -- Default. Stores chats in a local SQLite file. Zero config, no external database needed. Best for development, demos, and single-user deployments.
  - `orchid_ai.persistence.postgres.OrchidPostgresChatStorage` -- PostgreSQL backend. Requires `pip install "orchid-ai[postgres]"` and a running PostgreSQL instance. Best for production, multi-user, and Docker deployments.
  - Custom backends: implement the `OrchidChatStorage` ABC and reference your class here.
- **`storage.dsn`** -- Database connection string. For SQLite: a file path (e.g. `"~/.orchid/chats.db"`, `"/data/chats.db"`). The directory is created automatically. For PostgreSQL: a full DSN (e.g. `"postgresql://user:pass@localhost:5432/orchid"`).

#### `tracing`

| YAML Key | Env Var | Default |
|----------|---------|---------|
| `tracing.langsmith_tracing` | `LANGSMITH_TRACING` | `false` |
| `tracing.langsmith_api_key` | `LANGSMITH_API_KEY` | `""` |
| `tracing.langsmith_project` | `LANGSMITH_PROJECT` | `"agents"` |

- **`tracing.langsmith_tracing`** -- Enable LangSmith tracing for observability. When `true`, all LangGraph executions (agent runs, tool calls, LLM completions) are logged to LangSmith for debugging and analysis. Must be configured **before** the graph is built (handled automatically at startup). Default `false` to avoid unintended data transmission.
- **`tracing.langsmith_api_key`** -- Your LangSmith API key. Required when `langsmith_tracing` is `true`. Obtain from the LangSmith dashboard.
- **`tracing.langsmith_project`** -- LangSmith project name where traces are grouped. Default `"agents"`. Use different project names to separate traces by environment (e.g. `"agents-dev"`, `"agents-prod"`).

---

### Complete Example (All Parameters)

**agents.yaml** -- every available parameter:

```yaml
version: "1"

# ── Defaults (inherited by all agents) ───────────────────────
defaults:
  llm:
    model: "gemini/gemini-2.5-flash"
    temperature: 0.2
  rag:
    k: 5
    enabled: true
    rag_ttl: 3600                    # 1 hour default cache for tool results

# ── Supervisor ───────────────────────────────────────────────
supervisor:
  assistant_name: "Travel Assistant"
  routing_system_prompt: |
    You are the routing brain. Analyze the user's message and decide
    which agent(s) should handle it. Consider agent descriptions carefully.
  synthesis_system_prompt: |
    You are the synthesis layer. Combine results from all agents into
    a single, coherent response for the user.
  sequential_advance_prompt: |
    The previous agent has completed its step. Based on its output,
    decide whether to advance to the next step or respond directly.

# ── Global guardrails ────────────────────────────────────────
guardrails:
  input:
    - type: prompt_injection
      fail_action: block
    - type: content_safety
      fail_action: block
    - type: max_length
      fail_action: block
      config:
        max_characters: 10000
  output:
    - type: pii_detection
      fail_action: redact
      config:
        entities: [email, phone, ssn, credit_card]

# ── Global built-in tools ────────────────────────────────────
tools:
  format_date:
    handler: "myapp.tools.dates.format_date"
    description: "Format a date string into a specified format"
    inject_to_rag: false             # results NOT cached (default)
    rag_ttl: null                    # use agent default (default)
    parameters:                      # optional — auto-extracted from function signature when omitted
      value:
        type: string
        description: "Date string to parse (ISO-8601 or common formats)"
        required: true
      fmt:
        type: string
        description: "Output format using strftime pattern"
        required: false
        default: "%Y-%m-%d"

  get_exchange_rate:
    handler: "myapp.tools.finance.get_exchange_rate"
    description: "Get current exchange rate between two currencies"
    inject_to_rag: true              # results cached in RAG
    rag_ttl: 600                     # override: 10 min (rates change often)
    parameters:
      from_currency:
        type: string
        description: "Source currency code (e.g. USD, EUR)"
        required: true
      to_currency:
        type: string
        description: "Target currency code (e.g. GBP, JPY)"
        required: true

  calculate_budget:
    handler: "myapp.tools.finance.calculate_budget"
    description: "Calculate travel budget from itemized costs"
    inject_to_rag: true              # results cached
    rag_ttl: null                    # use agent default (3600s from defaults)

# ── Orchestrator-level skills (cross-agent) ──────────────────
skills:
  trip_planner:
    description: >
      Plan a complete trip: find flights, book hotels,
      and suggest activities at the destination.
    steps:
      - agent: flights
        instruction: "Search for flights to the destination on the requested dates"
      - agent: hotels
        instruction: "Based on the flight results, find hotels near the airport for those dates"
      - agent: activities
        instruction: "Suggest activities and restaurants at the destination for the trip duration"

  budget_review:
    description: >
      Review all booked items and produce a complete budget breakdown.
    steps:
      - agent: flights
        instruction: "Get the price summary for the booked flights"
      - agent: hotels
        instruction: "Get the price summary for the booked hotels"

# ── Agents ───────────────────────────────────────────────────
agents:

  # ── Agent with MCP servers + all MCP options ───────────────
  flights:
    description: >
      Flight search and booking agent. Searches airlines,
      compares prices, and can hold reservations.
    prompt: |
      You are a Flight Search Agent.
      Use the available tools to find and compare flights.
      Always present options sorted by price.
      Include airline, departure/arrival times, and layovers.

    # Per-agent LLM override
    llm:
      model: "openai/gpt-4o"
      temperature: 0.1

    # Per-agent RAG settings
    rag:
      namespace: flights
      k: 10                          # retrieve more results for flights
      enabled: true
      rag_ttl: 7200                  # 2 hour cache for this agent

    # MCP server connections
    mcp_servers:
      # Server with explicit tool allow-list
      - name: airline-api
        type: remote
        transport: streamable_http
        url: "${AIRLINE_MCP_URL}"
        tool_call_strategy: sequential
        tools:
          - name: search_flights
            arguments:
              currency: USD
            inject_to_rag: true      # cache search results
            rag_ttl: 1800            # override: 30 min for flight searches
          - name: hold_reservation
            inject_to_rag: false     # never cache booking actions
          - name: get_seat_map
            arguments:
              class: economy
            inject_to_rag: true      # cache seat maps
            rag_ttl: null            # use agent rag_ttl (7200s)
        prompts: []
        resources: []

      # Server with wildcard discovery (all tools + prompts)
      - name: price-tracker
        type: local
        transport: streamable_http
        url: "http://localhost:3002"
        tool_call_strategy: all
        tools: "*"                   # discover all tools at runtime
        prompts: "*"                 # discover all prompts at runtime
        resources: "*"               # discover all resources at runtime

    # Built-in tools available to this agent
    tools:
      - format_date
      - get_exchange_rate

    # Agent-level skills (multi-step workflows within this agent)
    skills:
      price_comparison:
        description: "Search multiple routes and compare prices side by side"
        steps:
          # Tool call step (MCP)
          - tool: search_flights
            source: airline-api
            arguments:
              max_results: 5
          # Tool call step (built-in)
          - tool: get_exchange_rate
            source: builtin
          # Agent invocation step (calls another agent directly)
          - agent: hotels
            instruction: "Find hotels near the destination airport for the same dates"

    # Per-agent guardrails (in addition to global)
    guardrails:
      input:
        - type: topic_restriction
          fail_action: warn
          config:
            allowed_topics: [flights, airlines, airports, travel, booking]

    execution_hints:
      parallel_safe: true

  # ── Agent with custom class ────────────────────────────────
  hotels:
    description: >
      Hotel search agent. Finds accommodations, compares ratings,
      and checks availability.
    prompt: |
      You are a Hotel Search Agent.
      Find the best hotel options based on location, dates, and budget.
      Prioritize ratings and proximity to landmarks.

    # Custom agent class (overrides GenericAgent)
    class: myapp.agents.hotels.HotelAgent

    rag:
      namespace: hotels
      k: 5
      enabled: true
      rag_ttl: 3600

    mcp_servers:
      - name: booking-api
        type: remote
        transport: sse             # SSE transport variant
        url: "${BOOKING_MCP_URL}"
        tool_call_strategy: llm_decides
        tools:
          - name: search_hotels
            inject_to_rag: true
          - name: check_availability
          - name: get_reviews
            inject_to_rag: true
            rag_ttl: 86400         # 24 hours for reviews (rarely change)
        prompts:
          - hotel_search_prompt
          - review_summary_prompt
        resources:
          - hotels://popular-destinations

    tools:
      - calculate_budget

    execution_hints:
      parallel_safe: true

  # ── Minimal agent (inherits all defaults) ──────────────────
  activities:
    description: >
      Activities and restaurant suggestion agent.
      Recommends things to do at the destination.
    prompt: |
      You are an Activities Agent.
      Suggest popular activities, restaurants, and experiences.
      Consider weather, season, and user preferences.

    # No LLM override (uses defaults: gemini/gemini-2.5-flash, temp 0.2)
    # No RAG override (uses defaults: enabled=true, k=5, rag_ttl=3600)
    # No MCP servers
    # No built-in tools
    # No skills

    rag:
      namespace: activities

    execution_hints:
      parallel_safe: false         # must run after other agents
```

**orchid.yml** -- every available parameter:

```yaml
# ── Agent config path ────────────────────────────────────────
agents:
  config_path: config/agents.yaml

# ── LLM providers ────────────────────────────────────────────
llm:
  model: gemini/gemini-2.5-flash
  ollama_api_base: http://localhost:11434
  groq_api_key: "gsk_..."
  gemini_api_key: "AIza..."
  anthropic_api_key: "sk-ant-..."
  openai_api_key: "sk-..."

# ── Authentication ───────────────────────────────────────────
auth:
  dev_bypass: false
  identity_resolver_class: "myapp.identity.MyIdentityResolver"
  domain: "myapp.example.com"

# ── Startup hook ─────────────────────────────────────────────
startup:
  hook: "myapp.startup.on_startup"

# ── RAG / Vector DB ──────────────────────────────────────────
rag:
  vector_backend: qdrant
  qdrant_url: http://localhost:6333
  embedding_model: text-embedding-3-small
  openai_api_key: "sk-..."
  gemini_api_key: "AIza..."

# ── Document upload ──────────────────────────────────────────
upload:
  vision_model: ollama/minicpm-v
  namespace: uploads
  max_size_mb: 20
  chunk_size: 1000
  chunk_overlap: 200

# ── Chat persistence ─────────────────────────────────────────
storage:
  class: orchid_ai.persistence.postgres.OrchidPostgresChatStorage
  dsn: postgresql://user:pass@localhost:5432/orchid

# ── Observability ────────────────────────────────────────────
tracing:
  langsmith_tracing: true
  langsmith_api_key: "lsv2_..."
  langsmith_project: "my-project"
```

## Guardrails

Orchid includes a 3-tier guardrail system that firewalls both the orchestrator and individual agents. Guardrails are configured entirely in YAML -- no code changes needed.

### Architecture

```
User message
  → Global input guardrails (prompt injection, content safety, max length, PII)
    → Supervisor routing
      → Per-agent input guardrails (topic restriction)
        → Agent execution
      → Per-agent output guardrails
    → Supervisor synthesis
  → Global output guardrails (PII redaction, groundedness)
→ Response
```

- **Global input guardrails** run on every user message before the supervisor sees it
- **Per-agent guardrails** run only when that specific agent is active
- **Global output guardrails** run on the final synthesized response

### Configuration

```yaml
# Global guardrails (apply to all agents)
guardrails:
  input:
    - type: prompt_injection
      fail_action: block
    - type: content_safety
      fail_action: block
    - type: max_length
      fail_action: block
      config:
        max_characters: 10000
    - type: pii_detection
      fail_action: redact
      config:
        entities: [credit_card, ssn]
  output:
    - type: pii_detection
      fail_action: redact
      config:
        entities: [email, phone, ssn, credit_card]

agents:
  basketball:
    description: "Basketball expert"
    prompt: "You are a basketball analyst."
    # Per-agent guardrails (in addition to global)
    guardrails:
      input:
        - type: topic_restriction
          fail_action: warn
          config:
            allowed_topics: [basketball, NBA, players, teams, stats]
```

### Built-in Guardrail Types

| Type | Purpose | Default Action |
|------|---------|---------------|
| `prompt_injection` | Detect instruction overrides, persona hijacks, delimiter injection | `block` |
| `content_safety` | Block harmful content (violence, self-harm, illegal activity) | `block` |
| `pii_detection` | Detect/redact emails, phones, credit cards, SSNs, IPs | `redact` |
| `max_length` | Reject messages exceeding a character limit | `block` |
| `topic_restriction` | Enforce per-agent domain boundaries via keyword matching | `warn` |
| `groundedness` | Check response grounding against RAG context | `warn` |

### Guardrail Actions

| Action | Behavior |
|--------|----------|
| `block` | Reject the message entirely; short-circuits the chain |
| `redact` | Replace matched content with `[REDACTED_<TYPE>]` placeholders; continues processing |
| `warn` | Allow the message but flag it in metadata |
| `log` | Silently log the detection; no user-visible effect |

### Custom Guardrails

Register custom guardrails by subclassing `Guardrail` and calling `register_guardrail()`:

```python
from orchid_ai import Guardrail, GuardrailContext, GuardrailResult, register_guardrail

class MyCustomGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "my_custom"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        if "forbidden" in content.lower():
            return GuardrailResult(
                triggered=True,
                action=self._fail_action,
                guardrail_name=self.name,
                message="Forbidden content detected.",
            )
        return GuardrailResult.passed(self.name)

register_guardrail("my_custom", MyCustomGuardrail)
```

Then use it in YAML:

```yaml
guardrails:
  input:
    - type: my_custom
      fail_action: block
```

## RAG Hierarchy

```
"__shared__"                 All tenants
  tenant_id                  All users in tenant
    user_id                  All user's chats
      chat_id
        scope="chat_shared"  All agents in chat
        scope="chat_agent"   Agent-private
```

Always use `OrchidRAGScope` — never raw `tenant_id` filters.

## Advanced features

### Mini-agents (parallel sub-task fork)

Opt-in per-agent block that turns a single supervisor turn into N
independent sub-tasks running in parallel through copies of the
parent agent.  Best for tool-heavy questions that decompose cleanly
("compare A and B and look up C") — the LangGraph builder synthesises
three nodes per opt-in agent (`{name}_agent`, `{name}_mini`,
`{name}_aggregator`) and the conditional edge fans out via `Send`.

```yaml
agents:
  research:
    description: "Multi-faceted research agent."
    prompt: "..."
    mini_agent:
      enabled: true                      # default: false
      max_count: 4                       # 2..8
      timeout_seconds: 60                # per-mini wall clock
      tool_allowlist_mode: strict        # strict | parent_full | inferred
      decomposer_model: gemini/gemini-flash-lite   # cheaper LLM for the splitter
      stream_inner_tokens: false         # surface only mini_agent.* events by default
      decomposer_prompt: |               # optional — overrides the default
        ...
      aggregator_prompt: |
        ...
      system_prompt_template: |          # optional — placeholders {parent_prompt}, {instruction}, {tool_list}
        {parent_prompt}

        === SUB-TASK ===
        {instruction}

        === TOOLS ===
        {tool_list}
```

Streaming consumers see `mini_agent.{decomposed,started,finished,aggregated}`
events.  Nesting is forbidden: child agents cannot enable
`mini_agent.enabled` (validation rejects it at config load).

### Parallel tool dispatch (`parallel_tools`)

Intra-round parallel dispatch.  When `parallel_tools: true`,
the agentic loop partitions one round's tool calls into:

- A **parallel batch** gathered via `asyncio.gather` — tools whose
  per-name `parallel_safe` is True.
- A **sequential tail** for everything else (HITL approvals, write
  effects, unknown safety).

`parallel_safe` resolves with this precedence (highest → lowest):

1. `requires_approval: true` → never parallel.
2. Built-in tool → `True` iff its top-level `tools.<name>.parallel_safe: true`.
3. MCP tool with explicit YAML `parallel_safe` → use it.
4. MCP tool without override → `True` iff the server advertised
   `readOnlyHint=true`.

Default `False` preserves today's serial behaviour.  See
[`examples/tool-strategies/`](../examples/tool-strategies/) for a
working demo.

### Internal prompt customisation

Every LLM-facing internal prompt is YAML-configurable with
backwards-compatible defaults.  Six surfaces exist; pick the
ones relevant to your deployment.

```yaml
# Top-level supervisor prompts.
supervisor:
  assistant_name: "Acme Knowledge Desk"
  routing_system_prompt: |
    You coordinate the Acme Knowledge Desk's specialist agents...
  synthesis_system_prompt: |
    You merge the specialists' outputs into a single answer...
  sequential_advance_prompt: |
    Hand off to the next specialist with a one-line summary of the prior step.
  history_summary_enabled: true       # sliding-window compression
  history_summary_recent_turns: 10

# Default RAG transformer prompts (inherited by all agents).
defaults:
  rag:
    retrieval:
      transformer_prompts:
        reformulate: |
          You rewrite ambiguous follow-ups into standalone search queries...

# Per-agent overrides — inherit from defaults; override granularly.
agents:
  legal_advisor:
    prompt: "..."
    prompt_sections:
      prior_results_header: "\n=== COUNSEL'S NOTES ==="
      mcp_prompt_template: "\n[authority {name}]\n{text}"
      rag_header: "\n=== SOURCE CITATIONS ==="
      resource_max_chars: 4000
      summarise_history_reminder: "\n\nFOCUS ON THE LATEST QUESTION."
      summarise_user_template: "Question: {query}\n\n{rag_section}Live data:\n{mcp_data}"
    rag:
      retrieval:
        strategy: hyde
        transformer_prompts:
          hyde:
            single: "Write one paragraph of plausible legal reasoning..."
            multi: "Write {n} legal-treatise paragraphs..."
          decompose: "Split into {n} legal sub-issues..."
```

Programmatic equivalents are exposed at:

- `OrchidAgentPromptConfig` (agentic-loop section templates + summarise overrides)
- `OrchidQueryTransformerPromptsConfig` (per-transformer prompts)
- `OrchidMiniAgentConfig.system_prompt_template` (per-mini focused prompt)
- `OrchidSupervisorConfig.routing_system_prompt` etc.

See [`examples/prompt-customization/`](../examples/prompt-customization/)
for an end-to-end example listing every override site.

### Custom retrieval strategies

`OrchidRetrievalStrategy` is a stateless ABC with `from_config` /
`retrieve` methods.  Subclass it, register at startup, then
reference by name in YAML.

```python
# my_pkg/strategies/recency.py
from orchid_ai.core.retrieval import OrchidRetrievalStrategy

class RecencyRetrieval(OrchidRetrievalStrategy):
    @classmethod
    def from_config(cls, config):
        return cls(field=getattr(config, "recency_field", "published_at"))

    def __init__(self, *, field):
        self._field = field

    async def retrieve(self, *, query, namespace, scope, k, reader, **_):
        results = await reader.retrieve(query=query, namespace=namespace, k=k * 2, scope=scope)
        results.sort(key=lambda r: r.document.metadata.get(self._field, 0), reverse=True)
        return results[:k]

# orchid.yml — register at startup
# startup:
#   hook: my_pkg.strategies.startup.register_strategies
```

Built-in strategies live under
`orchid_ai/rag/strategies/{simple,multi_query,hyde,hybrid,graph_rag}.py`
— short, readable templates for your own.

[`examples/rag-strategies/`](../examples/rag-strategies/) shows a full
custom strategy with a YAML side-by-side comparison.

### Custom tool-call strategies

`OrchidToolCallStrategy` controls how an MCP server's tools are
dispatched during **skill execution**.  Built-ins: `all`,
`sequential`, `llm_decides`.  Register custom strategies via
`register_strategy()` from a startup hook.

```python
from orchid_ai.agents.strategies import OrchidToolCallStrategy

class PriorityStrategy(OrchidToolCallStrategy):
    """Try tools in order; stop at the first non-empty result."""
    async def execute(self, client, tools, query, auth, *, agent_name="", **_):
        results = {}
        for tool in tools:
            r = await client.call_tool(tool.name, {"query": query, **tool.arguments}, auth)
            results[tool.name] = r.text
            if r.text.strip():
                break
        return results
```

```yaml
# Reference by name in agents.yaml
agents:
  cascade_lookup:
    mcp_servers:
      - name: kb
        url: ${KB_MCP_URL}
        tool_call_strategy: priority
        tools:
          - { name: cache_lookup }
          - { name: primary_lookup }
          - { name: slow_lookup }
```

Note: `tool_call_strategy` only fires inside skill-execution paths.
The default agentic loop (LLM picks tools via `tool_calls`) is always
"LLM decides" — see
[`examples/tool-strategies/`](../examples/tool-strategies/) for a
worked demo.

### Custom storage backends

Implement `OrchidChatStorage` and reference its dotted import path
in `orchid.yml`.  Constructor must accept `dsn=` and
`extra_migrations_package=` (the framework factory passes both
unconditionally).

```yaml
storage:
  class: my_pkg.storage.redis.OrchidRedisChatStorage
  dsn: redis://localhost:6379/0
```

The library ships SQLite (default) and PostgreSQL backends.  See
[`examples/custom-storage/`](../examples/custom-storage/) for a
JSON-file backend with the full contract checklist.

### Sliding-window history summarisation

For long-running chats the supervisor's history budget can blow past
the LLM's context window.  Opt in via
`supervisor.history_summary_enabled: true` and the framework keeps
the most recent `history_summary_recent_turns` (default 10) verbatim
while summarising older exchanges via a cheaper LLM
(`history_summary_model`, defaults to the supervisor model).
Compression runs only when the chat actually exceeds the recent-turn
threshold so short chats pay nothing.

### Pollen + Bloom (event-driven activation)

The `events:` YAML block (see [agents.yaml Reference → `events`](#events-pollen--bloom--optional-opt-in)) wires an opt-in async substrate that turns webhooks, cron schedules, and in-graph `emit_signal` calls into background LangGraph runs.

**Naming.** *Pollen* is the signal substrate (ingest → persist → enqueue). *Bloom* is the execution layer (dequeue → match trigger → run agent under a synthesised auth context). A `JobRun` is the unit of execution.

**The flow.**

```
   ┌──────────────┐  ingest  ┌────────────────────────┐    enqueue    ┌──────────────┐
   │ Producer     │ ───────▶ │ OrchidSignalDispatcher │ ────────────▶ │ Signal Queue │
   │ (HTTP/cron/  │          │ (persist + enqueue,    │   (atomic     │ (durable     │
   │  internal)   │          │  one transaction)      │    outbox)    │  buffer)     │
   └──────────────┘          └────────────────────────┘               └──────┬───────┘
                                                                              │
                                                                  drain      ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ AsyncioWorkerPoolProcessor                                                 │
   │   1. lease a Signal                                                         │
   │   2. resolve identity claim → OrchidAuthContext (via OrchidIdentityResolver)│
   │   3. find matching triggers (JMESPath ``when:`` evaluated here)             │
   │   4. insert a JobRun row, lock by parallelism_key, run GraphJobRunner       │
   │   5. on success / failure → emit BloomEvent stream events                   │
   └────────────────────────────────────────────────────────────────────────────┘
```

**Three identity flavours** for *who* the Bloom runs as (see `events.triggers[].emits.identity` in the YAML reference):

- `service_account` — named platform identity (e.g. `digest-bot`), no user-of-record.
- `addressed_to_user` — service identity tagged with a user id extracted from the signal (user-scoped RAG without impersonation).
- `act_as_user` — full user impersonation via `OrchidIdentityResolver.mint_for_user(tenant_key, user_id)`. Probed at boot.

**Chat binding (opt-in).** A signal MAY carry a `ChatBinding {chat_id, mode, on_failure, source_message_id?}`. When the matched trigger has `respect_chat_binding: true` AND the resolved auth has write permission on the target chat, the run's final `AIMessage` is appended to that chat with `metadata.origin="bloom"`. Cross-user smuggling is rejected at run time regardless of what the signal carried — the runner re-validates ownership through the resolved auth. `OrchidAgent.emit_signal(chat_id="self", ...)` auto-fills `source_message_id` so the frontend can anchor an in-chat live-progress card under the user message that produced the binding.

**`OrchidAgent.emit_signal`** is the in-graph hook for fan-out: an agent emits a signal that a separate trigger picks up to run a different agent. Internal emissions go through `dispatcher.ingest` — there is **no** in-process fast path that bypasses persistence, so internal Blooms get the same idempotency, retries, and visibility filtering as webhook-driven ones.

**Idempotency by construction.** `UNIQUE (source, dedupe_key)` on signals; `UNIQUE (trigger_id, signal_id, attempt_number)` on `job_runs`. Retries become new `JobRun` rows — never in-place updates.

**Streaming.** `BloomEventStream` is an in-process channel-keyed pub/sub (used by the orchid-api SSE endpoints):

- `run:{run_id}` channel — operator-grade trace: `bloom.run.queued`, `bloom.run.started`, `bloom.run.finished`, plus tool / agent ticks.
- `chat:{chat_id}` channel — chat-bound runs publish a redacted `ChatBloomEvent` stream: `chat.bloom.attached`, `chat.bloom.tick`, `chat.bloom.finished` (no raw tool result bodies, no run `result` payload — the final `AIMessage` flows through chat reload).

**Visibility.** `events.triggers[].emits.visibility` (and the resolved value carried on `JobSpec` / `JobRun`) drives a §26 visibility filter applied to every `SELECT FROM job_runs` / `signals` query in the API. Cross-tenant access is always rejected, even for admins. The reserved role string `OrchidAuthContext.roles = frozenset({"admin"})` unlocks the `admin` visibility level.

**External buses.** `RelayingSignalQueue` is a publish-then-mark adapter: the dispatcher persists with `relay_status=pending_publish`, the queue tries to publish to your `BusPublisher`, and `RelayRecoveryProducer` periodically sweeps pending rows so a transient publisher outage doesn't lose signals.

This whole layer is fully opt-in: omit `events:` (or set `events.enabled: false`) and zero new objects are constructed. See [`orchid_ai/events/AGENTS.md`](orchid_ai/events/AGENTS.md) for the package-level architecture rules and the boundary contract between `core/events/` (zero deps) and the concrete implementations.

### MCP capability cache warming

The first agentic round normally needs an MCP `tools/list` /
`prompts/list` / `resources/list` round-trip per server.
`OrchidSessionWarmer` proactively populates the cache:

- `auth.mode: none` servers warm at process startup
  (`Orchid.warm_unauthenticated_capabilities()`).
- `passthrough` and `oauth` servers warm at user-session start —
  the frontend calls `POST /session/warm` after login, with a
  fire-and-forget backstop on the first agentic loop.

Manual flush via `OrchidMCPClient.invalidate_cache()` or
`OrchidSessionWarmer.invalidate_user(auth)`.

## Embedding Dimensions

| Model | Dimensions |
|-------|-----------|
| ollama/nomic-embed-text | 768 |
| text-embedding-3-small | 1536 |
| gemini/gemini-embedding-001 | 3072 |

Switching models requires wiping and re-indexing Qdrant collections.

## MCP gateway exposure (optional)

`OrchidAgentsConfig` includes an **optional** `mcp_gateway` field that
lets integrators customise how Orchid is presented to MCP clients via
the `orchid-mcp` gateway — tool title/description overrides + MCP
Prompt templates. The block is entirely optional — empty by default,
ignored when not populated.

```yaml
# agents.yaml
mcp_gateway:
  tools:
    orchid_ask:
      title: "Ask the Acme Knowledge Base"
      description: "Route questions to the Acme support agents."
    # The Pollen + Bloom event tools (orchid_signal_emit /
    # orchid_bloom_status / orchid_bloom_list) override identically
    # — add an entry per tool the gateway exposes.
    orchid_signal_emit:
      title: "Trigger a background workflow"
      description: "Emit a Pollen signal to start an event-driven Bloom run."
  prompts:
    - name: compliance_report
      description: "Generate a compliance-completion report."
      arguments:
        - { name: department, required: true }
      template: |
        Produce a compliance report for {{department}}.
```

Exposed via `orchid-api`'s `GET /mcp-gateway/config` endpoint. Env-var
overrides (`ORCHID_MCP_GATEWAY_TOOL_*`, `ORCHID_MCP_GATEWAY_PROMPTS_FILE`)
live in `orchid-api`, not here.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -x          # all tests
pytest -k "test_scopes"   # specific
ruff check orchid/        # lint
ruff format orchid/       # format
```

## Code Style

- Python 3.11+, Ruff, line length 120
- `from __future__ import annotations` in every file
- Imports: `from orchid_ai.xxx` (never `from src.xxx`)

## License

MIT -- see [LICENSE](LICENSE).
