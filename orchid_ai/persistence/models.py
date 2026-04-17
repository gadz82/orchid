"""Data models for chat session persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC ``datetime`` — replacement for the deprecated ``datetime.utcnow()``.

    **Always treat the return value as UTC**; the ``tzinfo`` is stripped
    only to preserve compatibility with the existing SQLite / PostgreSQL
    columns, which were created before :class:`datetime.utcnow` was
    deprecated and accept naive values.

    Tech debt
    ---------
    Migrating to timezone-aware datetimes (and ``TIMESTAMPTZ`` columns on
    PostgreSQL) is the right long-term fix; everything calls this single
    helper, so that migration is a one-file change followed by a schema
    revision.  Tracked separately from the ``datetime.utcnow`` sweep.

    For new code not tied to the legacy schema, prefer
    ``datetime.now(timezone.utc)`` directly.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)
