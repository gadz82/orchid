"""Data models for chat session persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatSession:
    """A conversation session belonging to a specific user/tenant."""

    id: str
    tenant_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_shared: bool = False


@dataclass
class ChatMessage:
    """A single message within a chat session."""

    id: str
    chat_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    agents_used: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
