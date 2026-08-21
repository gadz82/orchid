"""MCP-server-related configuration models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema_rag import OrchidRAGConfig


class OrchidToolConfig(BaseModel):
    """A single MCP tool available to an agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    inject_to_rag: bool = False  # opt-in: store this tool's results in RAG
    rag_ttl: int | None = None  # per-tool TTL override; None = use agent default
    requires_approval: bool = False  # pause and ask user before executing (HITL)
    # Parallel tool-call dispatch override.  ``None`` means
    # "fall back to the MCP ``readOnlyHint`` annotation"; explicit
    # ``True`` / ``False`` wins over the annotation.  Only consulted
    # when the agent has ``parallel_tools: true`` set.
    parallel_safe: bool | None = None
    #: Per-tool RAG override.  When set, this tool's
    #: ingestion / retrieval / namespace / payload-index decisions
    #: use the tool's ``rag`` block instead of the agent's.  ``None``
    #: (the default) means inherit from the agent.  See
    #: :meth:`OrchidAgentConfig.effective_rag` for the merge contract.
    rag: OrchidRAGConfig | None = None


class OrchidMCPManualRegistrationConfig(BaseModel):
    """Manual OAuth seeding for a non-MCP-2025-03-26-compliant MCP server.

    Carries the authorization-server endpoints + client credentials that the
    framework would normally discover at runtime via the three-RFC chain
    (RFC 9728 → RFC 8414 → RFC 7591).  Only used when the target server does
    not advertise ``resource_metadata`` in its 401 response (e.g. Atlassian
    Rovo).  Kept as a separate, explicitly-named model so the canonical
    ``OrchidMCPAuthConfig`` surface stays minimal (``mode`` only) and YAML
    never silently carries static client credentials on the happy path.
    """

    model_config = ConfigDict(extra="forbid")

    authorization_endpoint: str = ""
    token_endpoint: str = ""
    registration_endpoint: str = ""
    client_id: str = ""
    client_secret: str = ""
    scopes: str = ""
    issuer: str = ""


class OrchidMCPAuthConfig(BaseModel):
    """Per-server authentication configuration.

    Determines how the MCP client authenticates with the server:

    - ``none`` (default): no authentication headers are sent.  Suitable
      for local MCP servers or remote servers that do not require auth.
    - ``passthrough``: forwards the graph's ``OrchidAuthContext`` bearer
      token unchanged — the MCP server shares the same identity provider
      as the API — the single-token-in-the-graph rule.
    - ``oauth``: the MCP server requires its own OAuth 2.0 flow.  The
      framework follows the MCP 2025-03-26 authorization spec and
      discovers everything at runtime from the server's 401 response:

        * Protected resource metadata (RFC 9728) — from the 401's
          ``WWW-Authenticate: Bearer resource_metadata="…"`` header.
        * Authorization server metadata (RFC 8414) — from
          ``/.well-known/oauth-authorization-server`` on the selected
          authorization server.
        * Dynamic client registration (RFC 7591) — POST'd to the
          advertised ``registration_endpoint``.

      For servers that do not implement MCP 2025-03-26 discovery (e.g.
      Atlassian Rovo), set ``manual_registration`` with the OAuth endpoints and
      credentials.  When present, the framework skips auto-discovery and uses
      those values directly.

    Example YAML (auto-discovery)::

        mcp_servers:
          - name: crm-backend
            url: https://crm.example.com/mcp
            auth:
              mode: oauth    # everything else is discovered at runtime

    Example YAML (manual seeding for non-compliant servers)::

        mcp_servers:
          - name: atlassian-rovo
            url: https://mcp.atlassian.com/v1/mcp
            auth:
              mode: oauth
              manual_registration:
                authorization_endpoint: https://auth.atlassian.com/authorize
                token_endpoint: https://auth.atlassian.com/oauth/token
                client_id: ${ATLASSIAN_CLIENT_ID}
                client_secret: ${ATLASSIAN_CLIENT_SECRET}
                scopes: "read:jira-work read:confluence-space.global"
    """

    mode: Literal["none", "passthrough", "oauth"] = "none"
    #: Opt-in manual OAuth seeding for servers that are NOT MCP 2025-03-26
    #: compliant (e.g. Atlassian Rovo, which returns a bare 401 without the
    #: required ``resource_metadata`` WWW-Authenticate parameter).  When set,
    #: the framework skips auto-discovery and uses these endpoints + creds
    #: directly.  ``None`` (default) means discovery is authoritative and
    #: YAML carries no static client credentials (per framework policy).
    manual_registration: OrchidMCPManualRegistrationConfig | None = None


class OrchidMCPServerConfig(BaseModel):
    """An MCP server connected to an agent.

    ``tools``, ``prompts``, and ``resources`` each accept either:
      - An explicit allow-list (e.g. ``["search_items", "create_record"]``)
      - The wildcard ``"*"`` to discover ALL capabilities from the server

    When any of them is ``"*"``, the corresponding ``discover_all_*`` flag is
    set and the allow-list is cleared — the agent will call ``list_tools()``,
    ``list_prompts()``, or ``list_resources()`` at runtime.

    Capability caches live for the process lifetime (driven by
    :class:`~orchid_ai.mcp.session_warmer.OrchidSessionWarmer`); flush
    them via :meth:`OrchidMCPClient.invalidate_cache`.  Unknown fields
    are rejected (``extra="forbid"``) so typos surface immediately.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["local", "remote"] = "local"
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    url: str  # supports ${ENV_VAR} interpolation (resolved by loader)
    auth: OrchidMCPAuthConfig = Field(default_factory=OrchidMCPAuthConfig)
    tools: list[OrchidToolConfig] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)  # prompt names to load (or "*")
    resources: list[str] = Field(default_factory=list)  # resource URIs/names to load (or "*")
    #: Name of an :class:`OrchidToolCallStrategy` registered via
    #: :func:`orchid_ai.agents.strategies.register_strategy`.  Built-ins
    #: are ``all`` / ``sequential`` / ``llm_decides``; integrators
    #: register custom strategies (e.g. ``priority``) from a startup
    #: hook before the first request hits.  Unknown names degrade to
    #: the ``all`` strategy at lookup time per
    #: :func:`get_strategy`'s safe-fallback contract.
    tool_call_strategy: str = "all"
    discover_all_tools: bool = False
    discover_all_prompts: bool = False
    discover_all_resources: bool = False

    @property
    def discover_all(self) -> bool:
        """True when ALL capabilities (tools + prompts + resources) are wildcarded."""
        return self.discover_all_tools and self.discover_all_prompts and self.discover_all_resources

    @model_validator(mode="before")
    @classmethod
    def _handle_wildcards(cls, data: Any) -> Any:
        """Convert ``"*"`` / ``["*"]`` into the corresponding ``discover_all_*`` flags."""
        if not isinstance(data, dict):
            return data

        for field, flag in [
            ("tools", "discover_all_tools"),
            ("prompts", "discover_all_prompts"),
            ("resources", "discover_all_resources"),
        ]:
            value = data.get(field)
            if value == "*" or value == ["*"]:
                data[field] = []
                data[flag] = True

        return data
