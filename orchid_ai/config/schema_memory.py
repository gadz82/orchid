"""Conversation memory configuration model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrchidMemoryConfig(BaseModel):
    """Configuration for conversation memory strategy.

    Controls how conversation context is persisted and retrieved
    beyond the current session's LangGraph state.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["none", "running_summary", "rag_augmented"] = "none"
    summary_recent_turns: int = Field(default=10, ge=1)
    summary_model: str | None = None
    summary_prompt: str | None = None
    persist_summary: bool = True
    structured_output: bool = True

    # ── RAG-augmented memory (Phase 3) ─────────────────────────

    rag_namespace: str = "__memory__"
    rag_k: int = Field(default=5, ge=1)
    rag_similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    store_turns: bool = True

    # ── Truncation (Phase 4) ───────────────────────────────────

    truncation_strategy: Literal["hard", "middle", "llm", "semantic"] = "hard"
    truncation_max_chars: int = Field(default=1000, ge=100)
