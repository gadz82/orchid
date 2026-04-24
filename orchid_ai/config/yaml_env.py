"""
Shared YAML-to-environment-variable mapping and application logic.

Used by both ``orchid-api/settings.py`` and ``orchid-cli/bootstrap.py``
to load ``orchid.yml`` values as environment variables (lowest priority —
real env vars always win).

Consumer-specific YAML keys (e.g. custom MCP server URLs) are simply
ignored — the mapping is strict and only processes known keys.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ── Nested YAML path → flat env var mapping ────────────────────
YAML_TO_ENV: dict[tuple[str, str], str] = {
    # ── agents ────────────────────────────────────────────────
    ("agents", "config_path"): "AGENTS_CONFIG_PATH",
    # ── llm ───────────────────────────────────────────────────
    ("llm", "model"): "LITELLM_MODEL",
    ("llm", "ollama_api_base"): "OLLAMA_API_BASE",
    ("llm", "groq_api_key"): "GROQ_API_KEY",
    ("llm", "gemini_api_key"): "GEMINI_API_KEY",
    ("llm", "anthropic_api_key"): "ANTHROPIC_API_KEY",
    ("llm", "openai_api_key"): "OPENAI_API_KEY",
    # ── auth ──────────────────────────────────────────────────
    ("auth", "dev_bypass"): "DEV_AUTH_BYPASS",
    ("auth", "identity_resolver_class"): "IDENTITY_RESOLVER_CLASS",
    ("auth", "auth_config_provider_class"): "AUTH_CONFIG_PROVIDER_CLASS",
    ("auth", "domain"): "AUTH_DOMAIN",
    ("auth", "oauth_client_id_env"): "AUTH_OAUTH_CLIENT_ID_ENV",
    ("auth", "oauth_scope"): "AUTH_OAUTH_SCOPE",
    # ── startup ──────────────────────────────────────────────
    ("startup", "hook"): "STARTUP_HOOK",
    # ── rag ───────────────────────────────────────────────────
    ("rag", "vector_backend"): "VECTOR_BACKEND",
    ("rag", "qdrant_url"): "QDRANT_URL",
    ("rag", "embedding_model"): "EMBEDDING_MODEL",
    ("rag", "openai_api_key"): "OPENAI_API_KEY",
    ("rag", "gemini_api_key"): "GEMINI_API_KEY",
    # ── upload ────────────────────────────────────────────────
    ("upload", "vision_model"): "VISION_MODEL",
    ("upload", "namespace"): "UPLOAD_NAMESPACE",
    ("upload", "max_size_mb"): "UPLOAD_MAX_SIZE_MB",
    ("upload", "chunk_size"): "CHUNK_SIZE",
    ("upload", "chunk_overlap"): "CHUNK_OVERLAP",
    # ── storage ───────────────────────────────────────────────
    ("storage", "class"): "CHAT_STORAGE_CLASS",
    ("storage", "dsn"): "CHAT_DB_DSN",
    ("storage", "extra_migrations_package"): "CHAT_EXTRA_MIGRATIONS_PACKAGE",
    # ── mcp_auth ──────────────────────────────────────────────
    ("mcp_auth", "token_store_class"): "MCP_TOKEN_STORE_CLASS",
    ("mcp_auth", "token_store_dsn"): "MCP_TOKEN_STORE_DSN",
    ("mcp_auth", "client_registration_store_class"): "MCP_CLIENT_REGISTRATION_STORE_CLASS",
    ("mcp_auth", "client_registration_store_dsn"): "MCP_CLIENT_REGISTRATION_STORE_DSN",
    # ── checkpointer ─────────────────────────────────────────
    ("checkpointer", "type"): "CHECKPOINTER_TYPE",
    ("checkpointer", "dsn"): "CHECKPOINTER_DSN",
    # ── api ───────────────────────────────────────────────────
    ("api", "base_url"): "API_BASE_URL",
    ("api", "cors_allowed_origins"): "CORS_ALLOWED_ORIGINS",
    ("api", "allow_index_endpoint"): "ALLOW_INDEX_ENDPOINT",
    # ── tracing ───────────────────────────────────────────────
    ("tracing", "langsmith_tracing"): "LANGSMITH_TRACING",
    ("tracing", "langsmith_api_key"): "LANGSMITH_API_KEY",
    ("tracing", "langsmith_project"): "LANGSMITH_PROJECT",
}


def apply_yaml_to_env(
    config_path: str,
    *,
    skip_sections: set[str] | None = None,
) -> int:
    """Load ``orchid.yml`` and export values as env vars (if not already set).

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    skip_sections : set[str] | None
        YAML sections to skip (e.g. ``{"storage"}`` for CLI which has
        its own defaults).

    Returns
    -------
    int
        Number of env vars applied.
    """
    try:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("[Config] YAML config %s not found — ignoring", config_path)
        return 0

    _skip = skip_sections or set()
    applied = 0

    for section, body in data.items():
        if not isinstance(body, dict) or section in _skip:
            continue
        for key, value in body.items():
            env_var = YAML_TO_ENV.get((section, key))
            if env_var is None:
                logger.debug("[Config] Unknown YAML key %s.%s — skipping", section, key)
                continue
            if env_var not in os.environ:
                os.environ[env_var] = str(value)
                applied += 1

    logger.info(
        "[Config] Loaded %d values from %s (env overrides take precedence)",
        applied,
        config_path,
    )
    return applied
