"""Customizable prompt fragments for the agentic loop and RAG transformers.

These models expose the previously-hardcoded prompt strings buried inside
:class:`GenericAgent` and the ``rag.transformers.*`` modules as YAML-
configurable settings, while preserving today's defaults verbatim so
existing deployments remain bit-identical.

Two blocks:

* :class:`OrchidAgentPromptConfig` — section templates assembled by
  ``GenericAgent._build_agentic_system_prompt`` (prior tool results,
  MCP prompts, MCP resources, RAG context).  Each template uses
  ``str.format``-style placeholders so customising one section does
  not force the integrator to rebuild the whole prompt.

* :class:`OrchidQueryTransformerPromptsConfig` — overrides for the
  four built-in query transformer prompts (``multi_query``, ``hyde``
  single + multi, ``decompose``, ``reformulate``).  ``None`` on any
  field means "use the module-level default", which is the same
  string the framework has always shipped.

The two blocks are intentionally separate — they are consumed by
different code paths (agent runtime vs RAG retrieval) and an
integrator typically wants to override one without touching the
other (ISP).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core.memory_types import (
    DEFAULT_NARRATIVE_FALLBACK_PROMPT,
    DEFAULT_STRUCTURED_EXTENSION_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_EXTENSION_USER_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT,
)

# ── Default templates ──────────────────────────────────────────

#: Header introducing the JSON dump of prior-turn tool results.
DEFAULT_PRIOR_RESULTS_HEADER = "\n--- Previous Tool Results (from prior turns) ---"

#: Template for a discovered MCP prompt that was rendered at discovery
#: time.  Placeholders: ``{name}``, ``{text}``.
DEFAULT_MCP_PROMPT_TEMPLATE = "\n--- MCP Prompt: {name} ---\n{text}"

#: Template for a discovered MCP prompt that requires arguments and
#: therefore could not be auto-rendered.  Placeholders: ``{name}``,
#: ``{description}``, ``{required_args}`` (comma-joined).
DEFAULT_SKIPPED_PROMPT_TEMPLATE = "\n[Available prompt: {name}] {description} (requires: {required_args})"

#: Header introducing the MCP resource block.
DEFAULT_RESOURCES_HEADER = "\n--- Available Resources ---"

#: Template for a single MCP resource.  Placeholders: ``{name}``,
#: ``{content}`` (already truncated to ``resource_max_chars``).
DEFAULT_RESOURCE_TEMPLATE = "\n[{name}]\n{content}"

#: Header introducing the RAG context block.
DEFAULT_RAG_HEADER = "\n--- Background Knowledge (RAG) ---"


# ── Summarise() helper defaults ────────────────────────────────

#: Reminder block appended to the summarise system prompt when
#: conversation history is present.
DEFAULT_SUMMARISE_HISTORY_REMINDER = (
    "\n\nIMPORTANT: The conversation history below shows prior exchanges. "
    "Always focus on the user's LATEST message and its relationship to "
    "the most recent topic. Do NOT change topic or introduce unrelated "
    "content unless the user explicitly asks for something new."
)

#: Header introducing the prior-tool-results JSON dump in the
#: summarise system prompt.  Distinct from
#: :data:`DEFAULT_PRIOR_RESULTS_HEADER` because the summarise path
#: prefixes "\n\n" while the agentic-loop builder uses "\n".
DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER = "\n\n--- Previous Tool Results (from prior turns) ---\n"

#: Header introducing the RAG block inside the summarise USER message.
#: Note the trailing newlines — the surrounding template concatenates
#: the JSON dump and the next section right after it.
DEFAULT_SUMMARISE_RAG_HEADER = "Background knowledge (from RAG):\n"

#: User-content template for the summarise call.  Placeholders:
#: ``{query}``, ``{rag_section}`` (already-rendered RAG block, may be
#: empty), and ``{mcp_data}`` (already-rendered JSON dump).
DEFAULT_SUMMARISE_USER_TEMPLATE = "User query: {query}\n\n{rag_section}Live data (from API):\n{mcp_data}"


class OrchidAgentPromptConfig(BaseModel):
    """Per-agent overrides for the agentic-loop system prompt assembly.

    All fields default to the module-level ``DEFAULT_*`` constants
    above.  Placeholders are resolved with :py:meth:`str.format`;
    integrators must keep every placeholder used by the corresponding
    default template (the framework asserts this at validation time
    when a custom value is supplied).
    """

    model_config = ConfigDict(extra="forbid")

    prior_results_header: str = DEFAULT_PRIOR_RESULTS_HEADER
    mcp_prompt_template: str = DEFAULT_MCP_PROMPT_TEMPLATE
    skipped_prompt_template: str = DEFAULT_SKIPPED_PROMPT_TEMPLATE
    resources_header: str = DEFAULT_RESOURCES_HEADER
    resource_template: str = DEFAULT_RESOURCE_TEMPLATE
    rag_header: str = DEFAULT_RAG_HEADER

    #: Maximum characters of the prior-tool-results JSON dump kept in
    #: the system prompt.  Anything beyond is truncated.
    prior_results_max_chars: int = Field(default=4000, ge=0)

    #: Maximum characters per MCP resource body kept in the system
    #: prompt.  Resources longer than this are truncated; raise this
    #: value when the agent depends on long technical artefacts.
    resource_max_chars: int = Field(default=2000, ge=0)

    # ── summarise() helper section overrides ───────────────────────
    #
    # The agent's ``summarise()`` method (used after the agentic loop
    # completes, or as a fallback when the loop produced no text)
    # composes its own system + user messages.  These four fields
    # expose its hardcoded fragments so the same per-agent overrides
    # mechanism applies to every LLM-facing surface, not just the
    # agentic-loop assembly.

    #: Reminder block appended to the summarise system prompt when
    #: ``conversation_history`` is non-empty.
    summarise_history_reminder: str = DEFAULT_SUMMARISE_HISTORY_REMINDER

    #: Header introducing the JSON dump of prior-turn tool results
    #: inside the summarise system prompt.
    summarise_prior_results_header: str = DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER

    #: Header introducing the RAG block inside the summarise USER
    #: message.
    summarise_rag_section_header: str = DEFAULT_SUMMARISE_RAG_HEADER

    #: User-content template for the summarise call — see
    #: :data:`DEFAULT_SUMMARISE_USER_TEMPLATE` for the placeholder
    #: contract.
    summarise_user_template: str = DEFAULT_SUMMARISE_USER_TEMPLATE

    #: Maximum characters of the prior-tool-results JSON dump kept in
    #: the summarise system prompt.
    summarise_prior_results_max_chars: int = Field(default=4000, ge=0)

    # ── Compression / conversation summary prompts ──────────────────
    #
    # Expose the prompts used by ``compress_conversation_history()`` and
    # ``OrchidInMemoryConversationMemory`` so integrators can customise
    # structured-extraction behaviour per agent.

    #: System prompt for structured summary extraction (flat → JSON).
    summary_compression_system_prompt: str = DEFAULT_STRUCTURED_SUMMARY_SYSTEM_PROMPT

    #: User prompt for structured summary extraction.
    summary_compression_user_prompt: str = DEFAULT_STRUCTURED_SUMMARY_USER_PROMPT

    #: System prompt when extending an existing structured summary.
    summary_extension_system_prompt: str = DEFAULT_STRUCTURED_EXTENSION_SYSTEM_PROMPT

    #: User prompt when extending an existing structured summary.
    summary_extension_user_prompt: str = DEFAULT_STRUCTURED_EXTENSION_USER_PROMPT

    #: Fallback prompt used when structured JSON extraction fails
    #: — produces a simple narrative summary instead.
    summary_narrative_fallback_prompt: str = DEFAULT_NARRATIVE_FALLBACK_PROMPT


# ── RAG transformer prompts ─────────────────────────────────────


class OrchidHydeTransformerPromptsConfig(BaseModel):
    """HyDE transformer prompts — separate single / multi templates.

    The transformer instantiates the ``single`` prompt when
    ``n_hypothetical=1`` and the ``multi`` prompt with the ``{n}``
    placeholder otherwise.  ``None`` means "use the built-in default".
    """

    model_config = ConfigDict(extra="forbid")

    single: str | None = None
    multi: str | None = None


class OrchidQueryTransformerPromptsConfig(BaseModel):
    """Per-transformer prompt overrides keyed by registry name.

    Each scalar field corresponds to a built-in transformer's system
    prompt.  ``None`` means "use the module-level default".
    Custom transformers registered via
    :func:`orchid_ai.rag.transformers.register_query_transformer` are
    free to ignore this block — the framework only forwards prompt
    kwargs to the four built-ins.
    """

    model_config = ConfigDict(extra="forbid")

    multi_query: str | None = None
    hyde: OrchidHydeTransformerPromptsConfig = Field(
        default_factory=OrchidHydeTransformerPromptsConfig,
    )
    decompose: str | None = None
    reformulate: str | None = None


__all__ = [
    "DEFAULT_MCP_PROMPT_TEMPLATE",
    "DEFAULT_PRIOR_RESULTS_HEADER",
    "DEFAULT_RAG_HEADER",
    "DEFAULT_RESOURCES_HEADER",
    "DEFAULT_RESOURCE_TEMPLATE",
    "DEFAULT_SKIPPED_PROMPT_TEMPLATE",
    "DEFAULT_SUMMARISE_HISTORY_REMINDER",
    "DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER",
    "DEFAULT_SUMMARISE_RAG_HEADER",
    "DEFAULT_SUMMARISE_USER_TEMPLATE",
    "OrchidAgentPromptConfig",
    "OrchidHydeTransformerPromptsConfig",
    "OrchidQueryTransformerPromptsConfig",
]
