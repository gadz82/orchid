"""
Shared state definitions for the LangGraph agent graph.

AuthContext is the identity envelope propagated to every agent and MCP client
(ADR-010 — Token Propagation).

The base AuthContext defines the **minimal contract** the framework needs:
  - ``access_token`` — for bearer passthrough to MCP servers
  - ``tenant_key``   — for RAG data isolation (multi-tenancy)
  - ``user_id``      — for chat ownership and user-scoped RAG
  - ``is_expired``   — for proactive token validation

Consumers subclass AuthContext to add platform-specific fields
(e.g. with ``installation_id``, ``domain``, ``paas_token``).
"""

from __future__ import annotations

from typing import Any, TypedDict


class AuthContext:
    """
    Base identity context — subclass to add platform-specific fields.

    The framework accesses ONLY these properties:
      - ``access_token``  — raw bearer token
      - ``bearer_header``  — ``{"Authorization": "Bearer <token>"}``
      - ``tenant_key``     — tenant identifier for RAG scoping (default: ``"default"``)
      - ``user_id``        — user identifier for chat ownership
      - ``is_expired``     — whether the token has expired

    Subclasses can override ``tenant_key`` and ``user_id`` to derive them
    from platform-specific fields (e.g. ``installation_id``).

    Example consumer subclass::

        class MyPlatformAuthContext(AuthContext):
            def __init__(self, *, access_token, domain, tenant_id,
                         user_uuid, **kwargs):
                super().__init__(access_token=access_token, **kwargs)
                self.domain = domain
                self._tenant_id = tenant_id
                self.user_uuid = user_uuid

            @property
            def tenant_key(self) -> str:
                return str(self._tenant_id)

            @property
            def user_id(self) -> str:
                return self.user_uuid
    """

    def __init__(
        self,
        *,
        access_token: str,
        tenant_key: str = "default",
        user_id: str = "",
        expires_at: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.access_token = access_token
        self._tenant_key = tenant_key
        self._user_id = user_id
        self.expires_at = expires_at
        self.extra = extra or {}

    # ── Framework contract (override in subclasses) ──────────

    @property
    def tenant_key(self) -> str:
        """
        Tenant identifier for RAG data isolation (ADR-014).

        Override in subclasses to derive from platform-specific fields.
        Returns ``"default"`` if not set.
        """
        return self._tenant_key or "default"

    @property
    def user_id(self) -> str:
        """
        User identifier for chat ownership and user-scoped RAG.

        Override in subclasses to derive from platform-specific fields.
        """
        return self._user_id

    @property
    def is_expired(self) -> bool:
        import time
        return self.expires_at > 0 and time.time() >= self.expires_at

    @property
    def bearer_header(self) -> dict[str, str]:
        """Ready-to-use Authorization header for MCP passthrough."""
        return {"Authorization": f"Bearer {self.access_token}"}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tenant_key={self.tenant_key!r}, "
            f"user_id={self.user_id!r}, "
            f"expired={self.is_expired})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AuthContext):
            return NotImplemented
        return (
            self.access_token == other.access_token
            and self.tenant_key == other.tenant_key
            and self.user_id == other.user_id
        )

    def __hash__(self) -> int:
        return hash((self.access_token, self.tenant_key, self.user_id))


# ── LangGraph State ─────────────────────────────────────────────
# NOTE: TypedDict is stdlib.  The `Annotated[list, add_messages]`
# annotation lives in the graph/ layer (which may import langgraph).
# Here we define the *shape* only.

class AgentState(TypedDict, total=False):
    """
    Canonical state schema for the LangGraph graph.

    `total=False` means every key is optional at construction time —
    the graph entry-point fills the required ones.
    """

    messages: list[Any]             # populated as Annotated[list, add_messages] in graph/
    auth_context: AuthContext       # ADR-010: one token for the whole session
                                    # ADR-014: tenant key = auth_context.tenant_key
    chat_id: str                    # chat session identifier for RAG scoping
    active_agents: list[str]
    mcp_context: dict[str, Any]     # raw data from MCP tool calls
    rag_context: dict[str, Any]     # chunks retrieved from vector store
    final_response: str | None
    skill_instructions: dict[str, str]  # ADR-017: agent_name → instruction from orchestrator skill
