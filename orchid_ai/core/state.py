"""
Shared state definitions for the LangGraph agent graph.

OrchidAuthContext is the identity envelope propagated to every agent and MCP client.

The base OrchidAuthContext defines the **minimal contract** the framework needs:
  - ``access_token`` — for bearer passthrough to MCP servers
  - ``tenant_key``   — for RAG data isolation (multi-tenancy)
  - ``user_id``      — for chat ownership and user-scoped RAG
  - ``is_expired``   — for proactive token validation

Consumers subclass OrchidAuthContext to add platform-specific fields
(e.g. with ``installation_id``, ``domain``, ``paas_token``).
"""

from __future__ import annotations

import time
from typing import Any, TypedDict


class OrchidAuthContext:
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

        class MyPlatformAuthContext(OrchidAuthContext):
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
        Tenant identifier for RAG data isolation.

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
        return self.expires_at > 0 and time.time() >= self.expires_at

    @property
    def bearer_header(self) -> dict[str, str]:
        """Ready-to-use Authorization header for MCP passthrough."""
        return {"Authorization": f"Bearer {self.access_token}"}

    # ── Persistence round-trip ──────────────────────────────
    #
    # Long-running clients (the orchid-cli, primarily) need to
    # cache an :class:`OrchidIdentityResolver` result on disk so
    # subsequent commands can reuse the resolved identity without
    # re-calling the upstream IdP every time.  The pair below
    # defines the contract:
    #
    #   ``to_storage_dict()``     — capture every field a caller
    #                               needs to recreate ``self``,
    #                               EXCLUDING ``access_token`` and
    #                               ``expires_at`` (those live in
    #                               the surrounding token record
    #                               and should be passed to
    #                               ``from_storage_dict()`` fresh
    #                               at restore time).
    #   ``from_storage_dict()``   — reconstruct an instance from
    #                               that dict + the always-fresh
    #                               ``access_token`` / ``expires_at``.
    #
    # Subclasses with typed attributes (e.g. a platform-specific
    # ``.domain`` / ``.tenant_uuid`` exposed alongside the base
    # contract) override BOTH methods to round-trip those attributes;
    # the default base-class implementation handles
    # ``tenant_key`` / ``user_id`` / ``extra`` only.

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialise non-secret state for the token store.

        The returned dict MUST be JSON-serialisable.  Do NOT include
        ``access_token`` or ``expires_at`` — those live one level up
        in the token record (where refresh rotates them) and are
        passed back into :meth:`from_storage_dict` as kwargs.
        """
        return {
            "tenant_key": self._tenant_key,
            "user_id": self._user_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_storage_dict(
        cls,
        *,
        access_token: str,
        expires_at: float,
        state: dict[str, Any],
    ) -> "OrchidAuthContext":
        """Reconstruct an instance from :meth:`to_storage_dict` output.

        ``access_token`` and ``expires_at`` are passed fresh from the
        surrounding token record (post-refresh, if applicable).
        """
        return cls(
            access_token=access_token,
            tenant_key=state.get("tenant_key", "default"),
            user_id=state.get("user_id", ""),
            expires_at=expires_at,
            extra=dict(state.get("extra") or {}),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tenant_key={self.tenant_key!r}, "
            f"user_id={self.user_id!r}, "
            f"expired={self.is_expired})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrchidAuthContext):
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


class OrchidAgentState(TypedDict, total=False):
    """
    Canonical state schema for the LangGraph graph.

    `total=False` means every key is optional at construction time —
    the graph entry-point fills the required ones.
    """

    messages: list[Any]  # populated as Annotated[list, add_messages] in graph/
    auth_context: OrchidAuthContext  # one token for the whole session
    # tenant key = auth_context.tenant_key
    chat_id: str  # chat session identifier for RAG scoping
    active_agents: list[str]
    mcp_context: dict[str, Any]  # raw data from MCP tool calls
    rag_context: dict[str, Any]  # chunks retrieved from vector store
    final_response: str | None
    skill_instructions: dict[str, str]  # agent_name → instruction from orchestrator skill
    _has_output_guardrails: bool  # sentinel for output guardrail routing
