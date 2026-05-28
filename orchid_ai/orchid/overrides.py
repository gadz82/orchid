"""Grouped config dataclasses to replace the 16-kwarg factory explosion.

M1 refactoring — each dataclass bundles related overrides that were
previously passed as individual keyword arguments to
``Orchid.from_config_path`` / ``Orchid.from_md_config``.

Backward-compat: the old 16-kwarg signatures are preserved as thin
wrappers that populate these dataclasses internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orchid_ai.core.content import OrchidContentSource


@dataclass
class StorageOverrides:
    """Chat persistence overrides."""

    chat_storage_class: str = ""
    chat_db_dsn: str = ""
    chat_extra_migrations_package: str | None = None


@dataclass
class MCPStorageOverrides:
    """MCP OAuth token + client-registration store overrides."""

    mcp_token_store_class: str = ""
    mcp_token_store_dsn: str = ""
    mcp_client_registration_store_class: str = ""
    mcp_client_registration_store_dsn: str = ""
    mcp_gateway_state_store_class: str = ""
    mcp_gateway_state_store_dsn: str = ""


@dataclass
class CheckpointerOverrides:
    """LangGraph checkpointer overrides."""

    checkpointer_type: str = ""
    checkpointer_dsn: str = ""


@dataclass
class StartupOverrides:
    """Startup hook overrides."""

    startup_hook: str = ""
    startup_hook_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchidFactoryOverrides:
    """Top-level bundle passed to ``Orchid.from_config_path``.

    Replaces the 16 individual keyword arguments.  The old kwargs are
    still accepted for backward compatibility — they populate this
    dataclass internally.
    """

    model: str = ""
    vector_backend: str = ""
    qdrant_url: str = ""
    embedding_model: str = ""
    storage: StorageOverrides = field(default_factory=StorageOverrides)
    mcp_storage: MCPStorageOverrides = field(default_factory=MCPStorageOverrides)
    checkpointer: CheckpointerOverrides = field(default_factory=CheckpointerOverrides)
    startup: StartupOverrides = field(default_factory=StartupOverrides)
    content_sources: list[OrchidContentSource] | None = None
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    skip_yaml_sections: set[str] = field(default_factory=set)
