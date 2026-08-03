"""Configuration model for external-agent CLI delegation tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrchidExternalAgentConfig(BaseModel):
    command: list[str] = Field(..., min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str = ""
    timeout: float = Field(default=600.0, gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    stdin_mode: Literal["arg", "stdin"] = "arg"
    normalizer: str = "passthrough"
    normalizer_instruction: str = ""
    normalizer_model: str = ""
    description: str = ""
    requires_approval: bool = True
    parallel_safe: bool = False
    inject_to_rag: bool = False
    rag_ttl: int | None = None
