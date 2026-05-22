"""Supervisor + execution-hint configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schema_memory import OrchidMemoryConfig


class ExecutionHints(BaseModel):
    """Hints for the Supervisor when routing."""

    parallel_safe: bool = True


class OrchidSupervisorConfig(BaseModel):
    """Supervisor prompt and behavior configuration.

    Allows consumers to customize the assistant name, prompts, and
    conversation history limits without modifying library code.
    When prompt fields are ``None``, the default templates in
    ``supervisor.py`` are used.

    History settings control how much prior conversation context is
    passed to the LLM during routing, synthesis, and sequential
    handoff steps:

    - ``history_max_turns``: maximum user/assistant exchange pairs
      to include.  Default 20 (= up to 40 messages).
    - ``history_max_chars``: maximum characters per individual
      message before truncation.  Default 1000.  Truncated messages
      get an ``…`` suffix.
    """

    assistant_name: str = "AI assistant"
    fallback_model: str | None = None  # fallback LLM for supervisor (overrides defaults.llm.fallback_model)
    streaming_enabled: bool = True  # enable SSE streaming for responses (default: on)
    routing_system_prompt: str | None = None
    synthesis_system_prompt: str | None = None
    sequential_advance_prompt: str | None = None
    history_max_turns: int = 20
    history_max_chars: int = 1000

    #: Optional cheaper / faster LLM used for the supervisor's routing
    #: + sequential-advance phases.  Those calls are short structured
    #: classifications and one-line handoff messages — they don't need
    #: the same model as the synthesis pass.  When ``None``, both
    #: phases reuse the supervisor's main ``chat_model``.  Example:
    #: ``routing_model: gemini/gemini-2.5-flash-lite``.
    routing_model: str | None = None

    # ── Sliding-window summarization (context compression) ──
    history_summary_enabled: bool = True
    history_summary_model: str | None = None  # None = use the supervisor model
    history_summary_recent_turns: int = 10  # keep last N turns verbatim

    # ── Synthesis fast-path ─────────────────────────────────
    #: When exactly one agent ran in the current turn and it produced a
    #: substantive final text response, return that text directly
    #: instead of running the supervisor synthesis LLM call.  Saves the
    #: cost of a redundant rewrite (typically 5–15 s on Gemini Flash
    #: with full conversation + tool-context injected).  Multi-agent
    #: turns and sequential pipelines still go through synthesis so
    #: their outputs can be merged.
    skip_synthesis_when_single_agent: bool = True

    #: Conversation memory strategy for incremental summarization.
    #: When ``strategy="running_summary"``, older conversation turns
    #: are incrementally extended rather than re-summarized from
    #: scratch on every turn (avoids O(n^2) LLM token waste).
    memory: OrchidMemoryConfig = Field(default_factory=OrchidMemoryConfig)
