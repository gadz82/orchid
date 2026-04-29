"""LLM-related configuration models."""

from __future__ import annotations

from pydantic import BaseModel


class OrchidLLMConfig(BaseModel):
    """LLM settings — can appear at defaults or agent level.

    The model string uses LiteLLM's ``provider/model-name`` format:
      - ``gemini/gemini-2.5-flash``          → Google AI Studio
      - ``groq/llama-3.3-70b-versatile``     → Groq
      - ``anthropic/claude-sonnet-4-20250514``  → Anthropic
      - ``ollama/llama3.2``                  → Local Ollama
      - ``openai/gpt-4o``                   → OpenAI

    The optional ``fallback_model`` is tried automatically when the
    primary model fails (503, rate limit, timeout).  When set at
    ``defaults.llm`` level, it applies to all agents and the supervisor
    unless overridden per-agent or per-supervisor.

    Example YAML::

        defaults:
          llm:
            model: gemini/gemini-2.5-flash
            fallback_model: ollama/llama3.2

        agents:
          critical-agent:
            llm:
              model: openai/gpt-4o
              fallback_model: anthropic/claude-sonnet-4-20250514
    """

    model: str = "gemini/gemini-2.5-flash"
    temperature: float = 0.2
    fallback_model: str | None = None
    retry_attempts: int = 0  # 0 = disabled; when > 0, transient errors retry with exponential backoff
