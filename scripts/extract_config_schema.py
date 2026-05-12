#!/usr/bin/env python3
"""
Extract the Orchid configuration schema into JSON.

USAGE
-----
Run from the orchid-website/ directory using the orchid venv:

    ../../orchid/.venv/bin/python scripts/extract_config_schema.py \\
        --out src/data/config-schema.json

Or if orchid-ai is installed in the active venv:

    python scripts/extract_config_schema.py --out src/data/config-schema.json

FILE MAPPING HEURISTIC
----------------------
  orchid.yml  : runtime / infrastructure keys from YAML_TO_ENV in
                orchid_ai.config.yaml_env (llm, rag, storage, auth, …).
  agents.yaml : every field reachable from OrchidAgentsConfig by recursive
                model walk (agents, supervisor, defaults, tools, skills, …).

The heuristic is deterministic and code-driven — there are no hand-maintained
lists.  Re-run whenever orchid_ai.config.schema or yaml_env changes.

EXAMPLES DETECTION
------------------
Each record's "examples" list is built by grepping every
  examples/**/{agents.yaml,orchid.yml}
under REPO_ROOT for the leaf key name (YAML-style key followed by ':').
File paths are mapped to /examples/<name> website routes via EXAMPLE_DIR_TO_ROUTE.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

# ── Bootstrap: add orchid package to sys.path ───────────────────────────────
# orchid-website/ lives at REPO_ROOT/orchid-website, so the orchid lib is at
# REPO_ROOT/orchid.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ORCHID_SRC = REPO_ROOT / "orchid"
if str(ORCHID_SRC) not in sys.path:
    sys.path.insert(0, str(ORCHID_SRC))

from pydantic import BaseModel  # noqa: E402
from pydantic.fields import FieldInfo  # noqa: E402
from pydantic_core import PydanticUndefined  # noqa: E402

from orchid_ai.config.schema import OrchidAgentsConfig  # noqa: E402
from orchid_ai.config.yaml_env import YAML_TO_ENV  # noqa: E402

# ── Example discovery ────────────────────────────────────────────────────────
EXAMPLES_DIR = REPO_ROOT / "examples"

EXAMPLE_DIR_TO_ROUTE: dict[str, str] = {
    "basketball": "/examples/basketball",
    "helpdesk": "/examples/helpdesk",
    "restaurant": "/examples/restaurant",
    "learning": "/examples/learning",
    "mcp-auth": "/examples/mcp-auth",
    "custom-storage": "/examples/custom-storage",
    "rag-strategies": "/examples/rag-strategies",
    "tool-strategies": "/examples/tool-strategies",
    "prompt-customization": "/examples/prompt-customization",
    "graph_kb": "/examples/graph-kb",
    "wiki": "/examples/wiki",
}


def _load_example_yamls() -> dict[str, str]:
    """Return {relative_path: content} for all example YAML files."""
    result: dict[str, str] = {}
    if not EXAMPLES_DIR.exists():
        return result
    for yaml_file in sorted(EXAMPLES_DIR.rglob("*.yaml")):
        rel = str(yaml_file.relative_to(REPO_ROOT))
        result[rel] = yaml_file.read_text(encoding="utf-8")
    for yml_file in sorted(EXAMPLES_DIR.rglob("*.yml")):
        rel = str(yml_file.relative_to(REPO_ROOT))
        result[rel] = yml_file.read_text(encoding="utf-8")
    return result


def _example_route(rel_path: str) -> str | None:
    parts = Path(rel_path).parts  # ('examples', 'basketball', 'agents.yaml')
    if len(parts) >= 2 and parts[0] == "examples":
        return EXAMPLE_DIR_TO_ROUTE.get(parts[1])
    return None


def _find_examples(
    leaf_key: str,
    yaml_files: dict[str, str],
    max_results: int = 3,
) -> list[str]:
    """Return /examples/* routes that contain 'leaf_key:' as a YAML key."""
    # Match the key as a YAML mapping key on any indentation level
    pattern = re.compile(rf"^\s*{re.escape(leaf_key)}\s*:", re.MULTILINE)
    routes: list[str] = []
    seen: set[str] = set()
    for rel_path, content in yaml_files.items():
        if not pattern.search(content):
            continue
        route = _example_route(rel_path)
        if route and route not in seen:
            seen.add(route)
            routes.append(route)
            if len(routes) >= max_results:
                break
    return routes


# ── Type introspection helpers ───────────────────────────────────────────────

def _is_pydantic_model(cls: Any) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel) and cls is not BaseModel
    except TypeError:
        return False


def _unwrap_optional(annotation: Any) -> Any | None:
    """If annotation is X | None or Optional[X], return X. Else None."""
    origin = get_origin(annotation)
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    )
    if is_union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return None


def _inner_list_type(annotation: Any) -> Any | None:
    """If list[X], return X. Else None."""
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def _inner_dict_value_type(annotation: Any) -> Any | None:
    """If dict[K, V], return V. Else None."""
    if get_origin(annotation) is dict:
        args = get_args(annotation)
        return args[1] if len(args) >= 2 else None
    return None


def _type_str(annotation: Any) -> str:  # noqa: PLR0911
    """Human-readable type string for a Pydantic annotation."""
    if annotation is None or annotation is type(None):
        return "null"
    if annotation is Any:
        return "any"

    origin = get_origin(annotation)
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    )
    if is_union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _type_str(non_none[0])
        return " | ".join(_type_str(a) for a in non_none)

    if origin is typing.Literal:
        return " | ".join(
            f'"{a}"' if isinstance(a, str) else str(a) for a in get_args(annotation)
        )

    if origin is list:
        args = get_args(annotation)
        inner = _type_str(args[0]) if args else "any"
        return f"list[{inner}]"

    if origin is dict:
        args = get_args(annotation)
        if len(args) >= 2:
            return f"dict[{_type_str(args[0])}, {_type_str(args[1])}]"
        return "dict"

    if _is_pydantic_model(annotation):
        return "object"

    primitives: dict[Any, str] = {str: "string", int: "int", float: "float", bool: "boolean"}
    if annotation in primitives:
        return primitives[annotation]

    return str(getattr(annotation, "__name__", annotation))


def _safe_default(field_info: FieldInfo) -> Any:
    """Return a JSON-safe default value, or None for required fields."""
    if field_info.default is not PydanticUndefined:
        d = field_info.default
        if isinstance(d, set):
            return sorted(d)
        return d

    factory = field_info.default_factory  # type: ignore[attr-defined]
    if factory is not None:
        try:
            val = factory()
        except Exception:
            return None
        if isinstance(val, BaseModel):
            return {}
        if isinstance(val, set):
            return []
        return val

    return None


def _is_required(field_info: FieldInfo) -> bool:
    has_default = field_info.default is not PydanticUndefined
    has_factory = field_info.default_factory is not None  # type: ignore[attr-defined]
    return not has_default and not has_factory


# ── Human-readable descriptions ──────────────────────────────────────────────
# Curated descriptions for the most commonly-used keys.  Keys map to dotted
# paths as they appear in the output (using [] for repeated-key collections).

DESCRIPTIONS: dict[str, str] = {
    # ── orchid.yml ──────────────────────────────────────────────
    "agents.config_path": "Path to the agents.yaml configuration file.",
    "llm.model": "Default LLM model in LiteLLM format (e.g. 'ollama/llama3.2', 'gemini/gemini-2.5-flash').",
    "llm.ollama_api_base": "Ollama server base URL for local model serving.",
    "llm.groq_api_key": "Groq API key for Groq-hosted models.",
    "llm.gemini_api_key": "Google AI (Gemini) API key.",
    "llm.anthropic_api_key": "Anthropic API key for Claude models.",
    "llm.openai_api_key": "OpenAI API key.",
    "auth.dev_bypass": "Bypass authentication — for development only; never set in production.",
    "auth.identity_resolver_class": "Dotted import path to an OrchidIdentityResolver subclass.",
    "auth.auth_config_provider_class": "Dotted import path to an OrchidAuthConfigProvider subclass.",
    "auth.auth_exchange_client_class": "Dotted import path to an OrchidAuthExchangeClient subclass.",
    "auth.domain": "Default domain used for identity resolution.",
    "auth.oauth_client_id_env": "Name of the env var holding the public OAuth client_id.",
    "auth.oauth_scope": "Advertised OAuth scope for downstream clients.",
    "startup.hook": "Dotted import path to a startup hook function called after graph init.",
    "rag.vector_backend": "Vector database backend — 'qdrant' is the only supported backend.",
    "rag.qdrant_url": "Qdrant server URL.",
    "rag.embedding_model": "Embedding model in LiteLLM format (e.g. 'ollama/nomic-embed-text').",
    "rag.openai_api_key": "OpenAI API key used by the embedding model.",
    "rag.gemini_api_key": "Google AI (Gemini) API key used by the embedding model.",
    "upload.vision_model": "Vision model for PDF/image OCR (e.g. 'ollama/minicpm-v').",
    "upload.namespace": "Qdrant namespace for uploaded documents.",
    "upload.max_size_mb": "Maximum upload size in megabytes.",
    "upload.chunk_size": "Text chunk size in characters for document ingestion.",
    "upload.chunk_overlap": "Character overlap between consecutive chunks.",
    "storage.class": "Dotted import path to an OrchidChatStorage subclass.",
    "storage.dsn": "Database connection string for chat persistence (SQLite path or Postgres URL).",
    "storage.extra_migrations_package": "Dotted package path for consumer-supplied DB migrations.",
    "mcp_auth.token_store_class": "Dotted import path to an OrchidMCPTokenStore subclass.",
    "mcp_auth.token_store_dsn": "Database DSN for per-user outbound MCP OAuth tokens.",
    "mcp_auth.client_registration_store_class": "Dotted import path to an OrchidMCPClientRegistrationStore subclass.",
    "mcp_auth.client_registration_store_dsn": "Database DSN for per-server MCP OAuth endpoints and DCR credentials.",
    "checkpointer.type": "LangGraph state persistence backend ('memory', 'sqlite', 'postgres', or class path).",
    "checkpointer.dsn": "Connection string or file path for the LangGraph checkpointer.",
    "api.base_url": "Public API URL used for OAuth callback construction.",
    "api.cors_allowed_origins": "Comma-separated list of allowed CORS browser origins.",
    "api.allow_index_endpoint": "Enable the POST /index admin endpoint for full reindexing.",
    "tracing.langsmith_tracing": "Enable LangSmith tracing for debugging and observability.",
    "tracing.langsmith_api_key": "LangSmith API key.",
    "tracing.langsmith_project": "LangSmith project name.",
    # ── agents.yaml — top level ─────────────────────────────────
    "version": "Schema version — only '1' is currently supported.",
    "defaults": "Default settings inherited by every agent unless overridden per-agent.",
    "agents": "Map of agent name → agent configuration. Every key becomes a routable agent.",
    "tools": "Global built-in tool declarations, keyed by tool name.",
    "skills": "Orchestrator-level cross-agent skill definitions.",
    "supervisor": "Supervisor prompt and behavior configuration.",
    "guardrails": "Global guardrail chains applied to every request.",
    "events": "Pollen + Bloom event-driven activation layer (opt-in).",
    # ── defaults.llm ──────────────────────────────────────────
    "defaults.llm.model": "Default LLM model for all agents (LiteLLM format).",
    "defaults.llm.temperature": "Sampling temperature (0.0–1.0). Lower = more deterministic.",
    "defaults.llm.fallback_model": "Model tried automatically when the primary model fails.",
    "defaults.llm.retry_attempts": "Retry count on transient LLM errors (0 = disabled).",
    # ── defaults.rag ──────────────────────────────────────────
    "defaults.rag.k": "Number of chunks retrieved per RAG query.",
    "defaults.rag.enabled": "Enable RAG context retrieval for all agents.",
    "defaults.rag.rag_ttl": "Cache TTL for RAG results in seconds (0 = no cache).",
    "defaults.rag.max_context_chars": "Maximum characters of RAG context injected into the system prompt.",
    "defaults.rag.ingestion.strategy": "Default chunking strategy: 'recursive', 'semantic', 'hierarchical', 'headered'.",
    "defaults.rag.ingestion.chunk_size": "Default text chunk size in characters.",
    "defaults.rag.ingestion.chunk_overlap": "Default character overlap between chunks.",
    "defaults.rag.retrieval.strategy": "Default retrieval strategy: 'simple', 'multi_query', 'hyde', 'hybrid', 'graph_rag'.",
    # ── supervisor ────────────────────────────────────────────
    "supervisor.assistant_name": "Display name for the orchestrating supervisor.",
    "supervisor.fallback_model": "Fallback LLM for the supervisor (overrides defaults.llm.fallback_model).",
    "supervisor.streaming_enabled": "Enable server-sent events (SSE) streaming for responses.",
    "supervisor.routing_system_prompt": "Custom system prompt for the supervisor's routing phase.",
    "supervisor.synthesis_system_prompt": "Custom system prompt for the synthesis phase.",
    "supervisor.sequential_advance_prompt": "Custom handoff prompt for sequential multi-agent flows.",
    "supervisor.history_max_turns": "Maximum conversation exchange pairs retained in context.",
    "supervisor.history_max_chars": "Maximum characters per message before truncation.",
    "supervisor.routing_model": "Cheaper model for routing and advance phases only.",
    "supervisor.history_summary_enabled": "Enable sliding-window summarization to compress older turns.",
    "supervisor.history_summary_model": "LLM model used for context summarization (None = supervisor model).",
    "supervisor.history_summary_recent_turns": "Number of recent turns kept verbatim during summarization.",
    "supervisor.skip_synthesis_when_single_agent": "Return a single agent's response directly without a synthesis LLM call.",
    # ── agents[] ──────────────────────────────────────────────
    "agents[].name": "Agent name (set automatically from the YAML dict key).",
    "agents[].description": "Human-readable purpose shown to the supervisor for routing.",
    "agents[].prompt": "System prompt injected into the agent's agentic loop.",
    "agents[].class": "Dotted import path to a custom OrchidAgent subclass.",
    "agents[].parallel_tools": "Dispatch independent read-only tool calls in parallel within one turn.",
    "agents[].llm.model": "Per-agent LLM model override.",
    "agents[].llm.temperature": "Per-agent sampling temperature.",
    "agents[].llm.fallback_model": "Per-agent fallback model.",
    "agents[].llm.retry_attempts": "Per-agent retry count on transient errors.",
    "agents[].rag.namespace": "Qdrant collection namespace for this agent.",
    "agents[].rag.k": "Number of chunks retrieved per RAG query for this agent.",
    "agents[].rag.enabled": "Enable RAG for this agent.",
    "agents[].rag.rag_ttl": "RAG cache TTL for this agent in seconds.",
    "agents[].rag.max_context_chars": "Maximum RAG context characters for this agent.",
    "agents[].rag.ingestion.strategy": "Chunking strategy for this agent.",
    "agents[].rag.ingestion.chunk_size": "Chunk size for this agent.",
    "agents[].rag.ingestion.chunk_overlap": "Chunk overlap for this agent.",
    "agents[].rag.retrieval.strategy": "Retrieval strategy for this agent.",
    "agents[].rag.retrieval.exclude_dynamic": "Exclude dynamically-injected tool output from retrieval.",
    "agents[].rag.retrieval.hyde.n_hypothetical": "Number of hypothetical answers generated per HyDE query.",
    "agents[].rag.retrieval.hybrid.sparse_encoder": "Sparse encoder for hybrid retrieval: 'bm25' or 'splade'.",
    "agents[].rag.retrieval.hybrid.fusion": "Fusion strategy for hybrid retrieval: 'rrf' or 'linear'.",
    "agents[].rag.retrieval.graph.enabled": "Enable graph-based retrieval for entity-relationship queries.",
    "agents[].rag.retrieval.graph.max_hops": "Maximum BFS depth for graph traversal.",
    "agents[].mcp_servers": "MCP servers this agent can call tools on.",
    "agents[].mcp_servers[].name": "Unique identifier for the MCP server.",
    "agents[].mcp_servers[].type": "Server type: 'local' (same host) or 'remote'.",
    "agents[].mcp_servers[].transport": "Transport protocol: 'streamable_http' or 'sse'.",
    "agents[].mcp_servers[].url": "MCP server URL. Supports ${ENV_VAR} interpolation.",
    "agents[].mcp_servers[].auth.mode": "Auth mode: 'none', 'passthrough', or 'oauth'.",
    "agents[].mcp_servers[].tool_call_strategy": "How tools are dispatched: 'all', 'sequential', 'llm_decides'.",
    "agents[].mcp_servers[].discover_all_tools": "Discover all tools from the server at runtime.",
    "agents[].mcp_servers[].discover_all_prompts": "Discover all prompts from the server at runtime.",
    "agents[].mcp_servers[].discover_all_resources": "Discover all resources from the server at runtime.",
    "agents[].execution_hints.parallel_safe": "Mark this agent safe to run in parallel with other agents.",
    "agents[].guardrails.input": "Per-agent input guardrail rules.",
    "agents[].guardrails.output": "Per-agent output guardrail rules.",
    "agents[].mini_agent.enabled": "Enable the mini-agent (self-clone) decomposition pattern.",
    "agents[].mini_agent.max_count": "Maximum number of parallel mini-agents (2–8).",
    "agents[].mini_agent.timeout_seconds": "Per-mini-agent timeout in seconds.",
    "agents[].mini_agent.tool_allowlist_mode": "Tool exposure mode: 'strict', 'parent_full', or 'inferred'.",
    "agents[].children": "Sub-agent configurations nested under this agent.",
    # ── tools[] ───────────────────────────────────────────────
    "tools[].handler": "Dotted import path to the Python function implementing this tool.",
    "tools[].description": "Tool description shown to the LLM for invocation decisions.",
    "tools[].inject_to_rag": "Store this tool's results in the RAG context store.",
    "tools[].rag_ttl": "Cache TTL for RAG-stored tool results (None = agent default).",
    "tools[].requires_approval": "Pause and prompt for human approval before executing (HITL).",
    "tools[].parallel_safe": "Declare the tool safe for parallel dispatch.",
    # ── skills[] ──────────────────────────────────────────────
    "skills[].description": "Human-readable skill purpose.",
    "skills[].steps": "Ordered list of agent invocations forming this cross-agent skill.",
    # ── guardrails ────────────────────────────────────────────
    "guardrails.input": "Guardrail rules applied to user input before any agent processes it.",
    "guardrails.output": "Guardrail rules applied to agent responses before delivery.",
    # ── defaults.rag.ingestion (continued) ───────────────────
    "defaults.rag.ingestion.parent_chunk_size": "Parent chunk size for hierarchical chunking (0 = disabled).",
    "defaults.rag.ingestion.parent_chunk_overlap": "Overlap for parent chunks in hierarchical chunking.",
    "defaults.rag.ingestion.post_processors": "Post-processors applied after chunking (e.g. 'contextual_headers', 'entity_extraction').",
    # ── defaults.rag.retrieval ────────────────────────────────
    "defaults.rag.retrieval.query_transformers": "Ordered list of query transformers (e.g. 'multi_query', 'hyde', 'reformulate').",
    "defaults.rag.retrieval.metadata_filters": "Metadata filter expressions applied to all retrievals.",
    "defaults.rag.retrieval.exclude_dynamic": "Exclude dynamically-injected tool output from retrieval results.",
    "defaults.rag.retrieval.hyde.n_hypothetical": "Number of hypothetical answers generated per HyDE query.",
    "defaults.rag.retrieval.hybrid.sparse_encoder": "Sparse encoder for hybrid retrieval: 'bm25' or 'splade'.",
    "defaults.rag.retrieval.hybrid.sparse_weight": "Weight of the sparse score in linear fusion (0.0–1.0).",
    "defaults.rag.retrieval.hybrid.fusion": "Fusion method for hybrid retrieval: 'rrf' (Reciprocal Rank Fusion) or 'linear'.",
    "defaults.rag.retrieval.hybrid.rrf_k": "RRF constant k (default 60, per Cormack et al.).",
    "defaults.rag.retrieval.graph.enabled": "Enable graph entity extraction and traversal for retrieval.",
    "defaults.rag.retrieval.graph.max_hops": "Maximum BFS depth from seed entities during graph traversal.",
    "defaults.rag.retrieval.graph.fuse_with_vectors": "Merge graph context with vector hits in the response.",
    "defaults.rag.retrieval.graph.relation_filter": "Restrict graph traversal to these edge labels (empty = all).",
    # ── agents[].rag (continued) ──────────────────────────────
    "agents[].rag.ingestion.parent_chunk_size": "Parent chunk size for hierarchical chunking (0 = disabled).",
    "agents[].rag.ingestion.parent_chunk_overlap": "Overlap for parent chunks.",
    "agents[].rag.ingestion.post_processors": "Post-processors applied after chunking.",
    "agents[].rag.retrieval.query_transformers": "Query transformer chain for this agent.",
    "agents[].rag.retrieval.metadata_filters": "Metadata filter expressions for this agent's retrievals.",
    "agents[].rag.retrieval.hyde.n_hypothetical": "Number of hypothetical answers for HyDE queries.",
    "agents[].rag.retrieval.hybrid.sparse_encoder": "Sparse encoder: 'bm25' or 'splade'.",
    "agents[].rag.retrieval.hybrid.sparse_weight": "Weight of sparse score in linear fusion.",
    "agents[].rag.retrieval.hybrid.fusion": "Fusion strategy: 'rrf' or 'linear'.",
    "agents[].rag.retrieval.hybrid.rrf_k": "RRF constant k.",
    "agents[].rag.retrieval.graph.enabled": "Enable graph-based retrieval.",
    "agents[].rag.retrieval.graph.max_hops": "Maximum BFS depth for graph traversal.",
    "agents[].rag.retrieval.graph.fuse_with_vectors": "Merge graph context with vector hits.",
    "agents[].rag.retrieval.graph.relation_filter": "Restrict graph traversal to these edge labels.",
    "agents[].rag.payload_indexes": "Explicit Qdrant payload index declarations (field → schema type).",
    # ── agents[].mcp_servers[].tools[] ────────────────────────
    "agents[].mcp_servers[].tools": "MCP tools this agent is allowed to call on this server.",
    "agents[].mcp_servers[].tools[].name": "MCP tool name to allow.",
    "agents[].mcp_servers[].tools[].arguments": "Default arguments passed to this tool.",
    "agents[].mcp_servers[].tools[].inject_to_rag": "Store this tool's results in the RAG context store.",
    "agents[].mcp_servers[].tools[].rag_ttl": "Per-tool RAG cache TTL (None = agent default).",
    "agents[].mcp_servers[].tools[].requires_approval": "Require human approval before this tool executes (HITL).",
    "agents[].mcp_servers[].tools[].parallel_safe": "Override parallel-safety for this tool.",
    "agents[].mcp_servers[].prompts": "MCP prompt names to load ('*' = discover all).",
    "agents[].mcp_servers[].resources": "MCP resource URIs to load ('*' = discover all).",
    # ── agents[].skills[] ─────────────────────────────────────
    "agents[].skills": "Per-agent skill definitions (multi-step workflows within this agent).",
    "agents[].skills[].description": "Human-readable skill description.",
    "agents[].skills[].steps": "Ordered steps: tool calls or agent invocations.",
    # ── agents[].mini_agent ───────────────────────────────────
    "agents[].mini_agent.decomposer_model": "LLM model for the mini-agent decomposer (None = agent model).",
    "agents[].mini_agent.stream_inner_tokens": "Stream individual mini-agent tokens to the SSE endpoint.",
    "agents[].mini_agent.decomposer_prompt": "Custom prompt for the decomposer step.",
    "agents[].mini_agent.aggregator_prompt": "Custom prompt for the aggregator step.",
    "agents[].mini_agent.system_prompt_template": "Template for each mini's system prompt ({parent_prompt}, {instruction}, {tool_list}).",
    # ── agents[].prompt_sections ──────────────────────────────
    "agents[].prompt_sections": "Custom templates for the agentic-loop system prompt assembly.",
    "agents[].prompt_sections.prior_results_header": "Header shown before the prior tool-results JSON block.",
    "agents[].prompt_sections.mcp_prompt_template": "Template for rendered MCP prompts.",
    "agents[].prompt_sections.resources_header": "Header shown before the MCP resources block.",
    "agents[].prompt_sections.resource_template": "Template for each MCP resource body.",
    "agents[].prompt_sections.rag_header": "Header shown before the RAG context block.",
    "agents[].prompt_sections.prior_results_max_chars": "Character cap on the prior tool-results JSON block.",
    "agents[].prompt_sections.resource_max_chars": "Character cap per MCP resource body.",
    # ── mcp_gateway ───────────────────────────────────────────
    "mcp_gateway": "MCP gateway exposure config — tool title/description overrides and MCP Prompts.",
    "mcp_gateway[].tools": "Tool title/description overrides keyed by MCP tool name.",
    "mcp_gateway[].prompts": "Pre-canned MCP Prompt templates exposed by the gateway.",
    # ── events ────────────────────────────────────────────────
    "events.enabled": "Enable the Pollen + Bloom event-driven activation layer.",
    "events.store": "Event storage backend configuration.",
    "events.queue": "Signal queue backend configuration.",
    "events.scheduler": "Scheduler backend (e.g. APScheduler) for cron-based triggers.",
    "events.producers": "List of signal producer configurations.",
    "events.processors": "List of signal processor / worker-pool configurations.",
    "events.schedules": "Cron schedule definitions.",
    "events.triggers": "Trigger definitions that map signals to agent activations.",
    "events.schedules[].id": "Unique schedule identifier.",
    "events.schedules[].cron": "Cron expression (e.g. '0 7 * * 1-5' for weekday 07:00 UTC).",
    "events.schedules[].trigger_id": "ID of the trigger this schedule fires.",
    "events.triggers[].id": "Unique trigger identifier.",
    "events.triggers[].emits.agent": "Agent to activate when this trigger fires.",
    "events.triggers[].emits.prompt_template": "Prompt template sent to the agent at activation.",
    "events.triggers[].retry.max": "Maximum retry attempts for failed trigger runs.",
    "events.triggers[].retry.backoff": "Retry backoff strategy: 'fixed', 'linear', or 'exponential'.",
    "events.triggers[].parallelism": "Parallelism mode: 'per_user', 'per_tenant', or 'unbounded'.",
}


# ── orchid.yml defaults (from Settings) ─────────────────────────────────────
# Hardcoded because Settings requires pydantic-settings and API-specific deps.

ORCHID_YML_DEFAULTS: dict[str, Any] = {
    "agents.config_path": "agents.yaml",
    "llm.model": "ollama/llama3.2",
    "llm.ollama_api_base": None,
    "llm.groq_api_key": None,
    "llm.gemini_api_key": None,
    "llm.anthropic_api_key": None,
    "llm.openai_api_key": None,
    "auth.dev_bypass": False,
    "auth.identity_resolver_class": None,
    "auth.auth_config_provider_class": None,
    "auth.auth_exchange_client_class": None,
    "auth.domain": None,
    "auth.oauth_client_id_env": None,
    "auth.oauth_scope": None,
    "startup.hook": None,
    "rag.vector_backend": "qdrant",
    "rag.qdrant_url": "http://qdrant:6333",
    "rag.embedding_model": "text-embedding-3-small",
    "rag.openai_api_key": None,
    "rag.gemini_api_key": None,
    "upload.vision_model": None,
    "upload.namespace": "uploads",
    "upload.max_size_mb": 20,
    "upload.chunk_size": 1000,
    "upload.chunk_overlap": 200,
    "storage.class": "orchid_ai.persistence.sqlite.OrchidSQLiteChatStorage",
    "storage.dsn": "~/.orchid/chats.db",
    "storage.extra_migrations_package": None,
    "mcp_auth.token_store_class": "orchid_ai.persistence.mcp_token_sqlite.OrchidSQLiteMCPTokenStore",
    "mcp_auth.token_store_dsn": "~/.orchid/chats.db",
    "mcp_auth.client_registration_store_class": (
        "orchid_ai.persistence.mcp_client_registration_sqlite"
        ".OrchidSQLiteMCPClientRegistrationStore"
    ),
    "mcp_auth.client_registration_store_dsn": "~/.orchid/chats.db",
    "checkpointer.type": None,
    "checkpointer.dsn": None,
    "api.base_url": "http://localhost:8000",
    "api.cors_allowed_origins": "http://localhost:3000,http://frontend:3000",
    "api.allow_index_endpoint": False,
    "tracing.langsmith_tracing": False,
    "tracing.langsmith_api_key": None,
    "tracing.langsmith_project": "agents",
}


# ── Schema builders ──────────────────────────────────────────────────────────

def _build_orchid_yml_entries(yaml_files: dict[str, str]) -> list[dict[str, Any]]:
    """Build config entries for orchid.yml from YAML_TO_ENV."""
    entries: list[dict[str, Any]] = []
    for (section, key), env_var in sorted(YAML_TO_ENV.items(), key=lambda x: x[0]):
        path = f"{section}.{key}"
        # Infer type from default value
        default = ORCHID_YML_DEFAULTS.get(path)
        if isinstance(default, bool):
            type_ = "boolean"
        elif isinstance(default, int):
            type_ = "int"
        elif isinstance(default, float):
            type_ = "float"
        else:
            type_ = "string"

        entries.append({
            "file": "orchid.yml",
            "path": path,
            "type": type_,
            "required": False,
            "default": default,
            "description": DESCRIPTIONS.get(path, f"Maps to environment variable {env_var}."),
            "deprecated": False,
            "examples": _find_examples(key, yaml_files),
        })
    return entries


def _walk_model(
    model_class: type[BaseModel],
    prefix: str,
    file: str,
    yaml_files: dict[str, str],
    visited: frozenset[str],
    depth: int = 0,
    max_depth: int = 6,
) -> list[dict[str, Any]]:
    """Recursively walk a Pydantic model, emitting one entry per field."""
    if depth > max_depth:
        return []
    model_key = f"{model_class.__qualname__}@{prefix}"
    if model_key in visited:
        return []
    visited = visited | {model_key}

    entries: list[dict[str, Any]] = []

    for field_name, field_info in model_class.model_fields.items():
        # Skip computed / serialization-excluded fields
        if getattr(field_info, "exclude", False):
            continue

        # Use alias as the effective YAML key when present
        yaml_key = field_info.alias if field_info.alias else field_name
        path = f"{prefix}.{yaml_key}" if prefix else yaml_key
        annotation = field_info.annotation

        # -- Unwrap Optional --
        unwrapped = _unwrap_optional(annotation)
        effective = unwrapped if unwrapped is not None else annotation

        # -- Decide how to handle the field --

        # Direct Pydantic model → recurse silently (model acts as a namespace)
        if _is_pydantic_model(effective):
            entries.extend(
                _walk_model(effective, path, file, yaml_files, visited, depth + 1, max_depth)
            )
            continue

        # list[PydanticModel] → emit list field + recurse into element type
        list_inner = _inner_list_type(effective)
        if list_inner is not None and _is_pydantic_model(list_inner):
            entries.append(_make_entry(field_name, field_info, path, file, yaml_files, f"list[object]"))
            entries.extend(
                _walk_model(list_inner, f"{path}[]", file, yaml_files, visited, depth + 1, max_depth)
            )
            continue

        # dict[str, PydanticModel] → emit dict field + recurse into value type
        dict_inner = _inner_dict_value_type(effective)
        if dict_inner is not None and _is_pydantic_model(dict_inner):
            entries.append(_make_entry(field_name, field_info, path, file, yaml_files, f"dict[string, object]"))
            entries.extend(
                _walk_model(dict_inner, f"{path}[]", file, yaml_files, visited, depth + 1, max_depth)
            )
            continue

        # Leaf type → emit record
        entries.append(_make_entry(field_name, field_info, path, file, yaml_files))

    return entries


def _make_entry(
    field_name: str,
    field_info: FieldInfo,
    path: str,
    file: str,
    yaml_files: dict[str, str],
    type_override: str | None = None,
) -> dict[str, Any]:
    """Build a single schema record."""
    annotation = field_info.annotation
    type_ = type_override or _type_str(annotation)
    default = _safe_default(field_info)
    required = _is_required(field_info)

    # Leaf key for example grepping (last segment, strip [] suffixes)
    leaf_key = path.split(".")[-1].rstrip("[]")
    examples = _find_examples(leaf_key, yaml_files)

    return {
        "file": file,
        "path": path,
        "type": type_,
        "required": required,
        "default": default,
        "description": DESCRIPTIONS.get(path, ""),
        "deprecated": False,
        "examples": examples,
    }


def _build_agents_yaml_entries(yaml_files: dict[str, str]) -> list[dict[str, Any]]:
    """Build config entries for agents.yaml from OrchidAgentsConfig."""
    return _walk_model(
        OrchidAgentsConfig,
        prefix="",
        file="agents.yaml",
        yaml_files=yaml_files,
        visited=frozenset(),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Orchid config schema to JSON.")
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    print("[extract_config_schema] Loading example YAML files…")
    yaml_files = _load_example_yamls()
    print(f"[extract_config_schema] Found {len(yaml_files)} example YAML files.")

    print("[extract_config_schema] Building orchid.yml schema…")
    orchid_yml_entries = _build_orchid_yml_entries(yaml_files)

    print("[extract_config_schema] Building agents.yaml schema…")
    agents_yaml_entries = _build_agents_yaml_entries(yaml_files)

    all_entries = orchid_yml_entries + agents_yaml_entries
    print(f"[extract_config_schema] Total entries: {len(all_entries)} "
          f"({len(orchid_yml_entries)} orchid.yml, {len(agents_yaml_entries)} agents.yaml)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, default=str)
        f.write("\n")

    print(f"[extract_config_schema] Written → {out_path}")


if __name__ == "__main__":
    main()
