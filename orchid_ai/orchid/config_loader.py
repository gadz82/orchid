"""OrchidConfigLoader — picks YAML / MD / hybrid path, returns config + runtime.

M1 refactoring: extracted from the 1281-LOC ``Orchid`` god module.
Owns the auto-detection logic (YAML vs Markdown vs hybrid) and delegates
to ``_build_runtime`` for the actual bootstrap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchid_ai.bootstrap import _build_runtime, _BootstrapResult
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.orchid.overrides import OrchidFactoryOverrides

logger = logging.getLogger(__name__)


@dataclass
class LoadedConfig:
    """Result of a config load — config + bootstrap result."""

    config: OrchidAgentsConfig
    bootstrap: _BootstrapResult
    config_file_hashes: dict[str, str] | None = None
    config_storage: Any | None = None
    config_watcher: Any | None = None


class OrchidConfigLoader:
    """Loads Orchid configuration from YAML, Markdown, or hybrid sources.

    Auto-detects the config format by file extension and delegates to the
    appropriate loader.  Returns a :class:`LoadedConfig` bundle.
    """

    @classmethod
    async def load(
        cls,
        config_path: str,
        *,
        apply_yaml: bool = True,
        agents_config_path: str = "",
        overrides: OrchidFactoryOverrides | None = None,
    ) -> LoadedConfig:
        """Load config with auto-detection of format.

        Parameters
        ----------
        config_path : str
            Path to ``orchid.yml`` (YAML) or ``orchid.md`` (Markdown).
            Auto-detected by file extension.
        apply_yaml : bool
            Skip the YAML → env step when ``False``.
        agents_config_path : str
            Path to ``agents.yaml``.  Ignored when auto-detection selects
            Markdown config.
        overrides : OrchidFactoryOverrides | None
            Grouped override dataclass.  ``None`` means use defaults.
        """
        o = overrides or OrchidFactoryOverrides()
        config_path_obj = Path(config_path) if config_path else None

        if config_path_obj is not None:
            # Pure Markdown config
            if config_path_obj.suffix == ".md":
                return await cls._load_md_config(
                    config_path,
                    agents_config_path=agents_config_path,
                    overrides=o,
                )

            # Directory containing orchid.md
            if config_path_obj.is_dir():
                md_candidate = config_path_obj / "orchid.md"
                if md_candidate.exists():
                    return await cls._load_md_config(
                        str(md_candidate),
                        agents_config_path=agents_config_path,
                        overrides=o,
                    )

            # Hybrid: YAML root + MD agents
            if config_path_obj.suffix in (".yml", ".yaml") and config_path_obj.exists():
                agents_dir = Path(agents_config_path or os.environ.get("AGENTS_CONFIG_PATH", "agents"))
                if not agents_dir.is_absolute():
                    agents_dir = config_path_obj.parent / agents_dir
                if agents_dir.is_dir() and any(agents_dir.glob("*.md")):
                    return await cls._load_hybrid_config(
                        orchid_yml_path=config_path_obj,
                        agents_dir=agents_dir,
                        apply_yaml=apply_yaml,
                        overrides=o,
                    )

        # Default: pure YAML config
        return await cls._load_yaml_config(
            config_path=config_path,
            apply_yaml=apply_yaml,
            agents_config_path=agents_config_path,
            overrides=o,
        )

    @classmethod
    async def _load_yaml_config(
        cls,
        *,
        config_path: str,
        apply_yaml: bool,
        agents_config_path: str,
        overrides: OrchidFactoryOverrides,
    ) -> LoadedConfig:
        """Standard YAML-only bootstrap."""
        result = await _build_runtime(
            config_path=config_path,
            apply_yaml=apply_yaml,
            config=None,
            agents_config_path=agents_config_path,
            model=overrides.model,
            vector_backend=overrides.vector_backend,
            qdrant_url=overrides.qdrant_url,
            embedding_model=overrides.embedding_model,
            chat_storage_class=overrides.storage.chat_storage_class,
            chat_db_dsn=overrides.storage.chat_db_dsn,
            chat_extra_migrations_package=overrides.storage.chat_extra_migrations_package,
            mcp_token_store_class=overrides.mcp_storage.mcp_token_store_class,
            mcp_token_store_dsn=overrides.mcp_storage.mcp_token_store_dsn,
            mcp_client_registration_store_class=overrides.mcp_storage.mcp_client_registration_store_class,
            mcp_client_registration_store_dsn=overrides.mcp_storage.mcp_client_registration_store_dsn,
            mcp_gateway_state_store_class=overrides.mcp_storage.mcp_gateway_state_store_class,
            mcp_gateway_state_store_dsn=overrides.mcp_storage.mcp_gateway_state_store_dsn,
            checkpointer_type=overrides.checkpointer.checkpointer_type,
            checkpointer_dsn=overrides.checkpointer.checkpointer_dsn,
            startup_hook=overrides.startup.startup_hook,
            startup_hook_kwargs=overrides.startup.startup_hook_kwargs,
            content_sources=overrides.content_sources,
            runtime_overrides=overrides.runtime_overrides,
            skip_yaml_sections=overrides.skip_yaml_sections or None,
        )

        # Merge DB-sourced configs if config storage is enabled
        config_storage = await cls._maybe_init_config_storage(result.config)

        return LoadedConfig(
            config=result.config,
            bootstrap=result,
            config_storage=config_storage,
        )

    @classmethod
    async def _load_md_config(
        cls,
        root_path: str,
        *,
        agents_config_path: str = "",
        watch: bool = True,
        overrides: OrchidFactoryOverrides | None = None,
    ) -> LoadedConfig:
        """Markdown config bootstrap (orchid.md + agents/*.md)."""
        o = overrides or OrchidFactoryOverrides()

        from orchid_ai.config.frontmatter import load_markdown_file
        from orchid_ai.config.md_loader import load_md_config, md_infrastructure_to_env

        root_path_as_path = Path(root_path)

        # Load MD config
        agents_dir_override = Path(agents_config_path) if agents_config_path else None
        config, file_hashes = load_md_config(root_path_as_path, agents_dir=agents_dir_override)

        # Apply infrastructure env vars
        root_md = load_markdown_file(root_path_as_path)
        env_vars = md_infrastructure_to_env(root_md.frontmatter, skip_sections=o.skip_yaml_sections or None)
        for k, v in env_vars.items():
            if k not in os.environ:
                os.environ[k] = v
        os.environ.setdefault("ORCHID_CONFIG", str(root_path_as_path))

        # Build runtime
        result = await _build_runtime(
            config_path=str(root_path_as_path),
            apply_yaml=False,
            config=config,
            model=o.model,
            vector_backend=o.vector_backend,
            qdrant_url=o.qdrant_url,
            embedding_model=o.embedding_model,
            chat_storage_class=o.storage.chat_storage_class,
            chat_db_dsn=o.storage.chat_db_dsn,
            chat_extra_migrations_package=o.storage.chat_extra_migrations_package,
            mcp_token_store_class=o.mcp_storage.mcp_token_store_class,
            mcp_token_store_dsn=o.mcp_storage.mcp_token_store_dsn,
            mcp_client_registration_store_class=o.mcp_storage.mcp_client_registration_store_class,
            mcp_client_registration_store_dsn=o.mcp_storage.mcp_client_registration_store_dsn,
            mcp_gateway_state_store_class=o.mcp_storage.mcp_gateway_state_store_class,
            mcp_gateway_state_store_dsn=o.mcp_storage.mcp_gateway_state_store_dsn,
            checkpointer_type=o.checkpointer.checkpointer_type,
            checkpointer_dsn=o.checkpointer.checkpointer_dsn,
            startup_hook=o.startup.startup_hook,
            startup_hook_kwargs=o.startup.startup_hook_kwargs,
            content_sources=o.content_sources,
            runtime_overrides=o.runtime_overrides,
            skip_yaml_sections=o.skip_yaml_sections or None,
        )

        # Set up config watcher for hot-reload
        config_watcher = None
        if watch:
            from orchid_ai.config.watcher import OrchidConfigWatcher

            agents_dir_resolved = cls._resolve_agents_dir_from_md(
                root_path_as_path, root_md.frontmatter, agents_dir_override
            )
            config_watcher = OrchidConfigWatcher(
                root_path=root_path_as_path,
                agents_dir=agents_dir_resolved,
                initial_config=config,
                initial_hashes=file_hashes,
            )

        return LoadedConfig(
            config=result.config,
            bootstrap=result,
            config_file_hashes=file_hashes if watch else None,
            config_watcher=config_watcher,
        )

    @classmethod
    async def _load_hybrid_config(
        cls,
        *,
        orchid_yml_path: Path,
        agents_dir: Path,
        apply_yaml: bool,
        overrides: OrchidFactoryOverrides,
    ) -> LoadedConfig:
        """Hybrid bootstrap: YAML root + MD agents."""
        import yaml

        from orchid_ai.config.md_loader import (
            _AGENT_BEHAVIOUR_FIELDS,
            _load_agents,
            build_config_data_from_yaml,
        )
        from orchid_ai.config.schema import OrchidAgentsConfig
        from orchid_ai.config.yaml_env import apply_yaml_to_env

        # Apply YAML infrastructure → env vars
        if apply_yaml and str(orchid_yml_path):
            os.environ.setdefault("ORCHID_CONFIG", str(orchid_yml_path))
            apply_yaml_to_env(str(orchid_yml_path), skip_sections=overrides.skip_yaml_sections or None)

        # Read orchid.yml for top-level agent config
        yaml_data: dict[str, Any] = {}
        try:
            with open(orchid_yml_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

        # Build top-level config data
        agent_configs, _ = _load_agents(agents_dir)
        config_data = build_config_data_from_yaml(yaml_data, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

        # Validate
        config = OrchidAgentsConfig.model_validate(config_data)

        # Build runtime with the pre-loaded config
        result = await _build_runtime(
            config_path=str(orchid_yml_path),
            apply_yaml=False,
            config=config,
            model=overrides.model,
            vector_backend=overrides.vector_backend,
            qdrant_url=overrides.qdrant_url,
            embedding_model=overrides.embedding_model,
            chat_storage_class=overrides.storage.chat_storage_class,
            chat_db_dsn=overrides.storage.chat_db_dsn,
            chat_extra_migrations_package=overrides.storage.chat_extra_migrations_package,
            mcp_token_store_class=overrides.mcp_storage.mcp_token_store_class,
            mcp_token_store_dsn=overrides.mcp_storage.mcp_token_store_dsn,
            mcp_client_registration_store_class=overrides.mcp_storage.mcp_client_registration_store_class,
            mcp_client_registration_store_dsn=overrides.mcp_storage.mcp_client_registration_store_dsn,
            mcp_gateway_state_store_class=overrides.mcp_storage.mcp_gateway_state_store_class,
            mcp_gateway_state_store_dsn=overrides.mcp_storage.mcp_gateway_state_store_dsn,
            checkpointer_type=overrides.checkpointer.checkpointer_type,
            checkpointer_dsn=overrides.checkpointer.checkpointer_dsn,
            startup_hook=overrides.startup.startup_hook,
            startup_hook_kwargs=overrides.startup.startup_hook_kwargs,
            content_sources=overrides.content_sources,
            runtime_overrides=overrides.runtime_overrides,
            skip_yaml_sections=overrides.skip_yaml_sections or None,
        )

        return LoadedConfig(
            config=result.config,
            bootstrap=result,
        )

    @classmethod
    async def _maybe_init_config_storage(
        cls,
        config: OrchidAgentsConfig,
    ) -> Any | None:
        """Initialize DB-backed config storage if enabled."""
        if not config.config_storage.enabled:
            return None

        from orchid_ai.config.storage_factory import build_config_storage

        cfg = config.config_storage
        config_store_instance = build_config_storage(cfg.class_path, cfg.dsn)
        await config_store_instance.init_db()
        db_configs = await config_store_instance.list_configs()
        if db_configs:
            config.merge_from_db(db_configs, strict=True)

        return config_store_instance

    @staticmethod
    def _resolve_agents_dir_from_md(
        root_path: Path,
        frontmatter: dict[str, Any],
        agents_dir_override: Path | None,
    ) -> Path:
        from orchid_ai.config.md_loader import _resolve_agents_dir

        return _resolve_agents_dir(root_path, frontmatter, agents_dir_override)
