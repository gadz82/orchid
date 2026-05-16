"""Markdown configuration loader.

Loads ``orchid.md`` + ``agents/*.md`` and produces a validated
:class:`OrchidAgentsConfig` plus an env-var mapping for infrastructure
keys — the same output shape as the existing YAML load path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .frontmatter import MarkdownFile, load_markdown_file
from .schema import OrchidAgentsConfig
from .yaml_env import YAML_TO_ENV

logger = logging.getLogger(__name__)

_AGENT_BEHAVIOUR_FIELDS: frozenset[str] = frozenset(OrchidAgentsConfig.model_fields.keys())


def _infer_agent_name(path: Path) -> str:
    return path.stem


def _merge_agent_md(md: MarkdownFile) -> dict[str, Any]:
    """Build an agent-config dict from a parsed MarkdownFile.

    The frontmatter holds structured fields (``description``, ``tools``,
    ``rag``, …); the Markdown body becomes the ``prompt``.  The
    ``class`` frontmatter key is passed through verbatim and is aliased
    to ``class_path`` by Pydantic during validation.
    """
    data: dict[str, Any] = dict(md.frontmatter)
    data["prompt"] = md.body
    return data


def _load_agents(agents_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Scan ``agents_dir`` for ``*.md`` files and build agent config dicts.

    Returns
    -------
    tuple[dict, dict]
        (agent_name → agent_config_data_dict, file_path_str → sha256).
        The second dict is for the watcher.
    """
    if not agents_dir.exists() or not agents_dir.is_dir():
        logger.warning("[MD Config] Agents directory not found: %s", agents_dir)
        return {}, {}

    agent_configs: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}

    for md_path in sorted(agents_dir.glob("*.md")):
        agent_name = _infer_agent_name(md_path)

        if agent_name in agent_configs:
            raise ValueError(
                f"Duplicate agent name '{agent_name}' from files "
                f"'{md_path}' and another in '{agents_dir}'. "
                f"Rename one of the files."
            )

        agent_md = load_markdown_file(md_path)
        agent_configs[agent_name] = _merge_agent_md(agent_md)
        file_hashes[str(md_path)] = agent_md.sha256

    logger.info(
        "[MD Config] Loaded %d agent(s) from %s: %s",
        len(agent_configs),
        agents_dir,
        list(agent_configs.keys()),
    )
    return agent_configs, file_hashes


def md_infrastructure_to_env(
    frontmatter: dict[str, Any],
    *,
    skip_sections: set[str] | None = None,
) -> dict[str, str]:
    """Map infrastructure keys from an MD frontmatter to environment variables.

    Uses the shared ``YAML_TO_ENV`` mapping.  Only processes keys that
    YAML_TO_ENV knows about; unknown keys are silently skipped.

    Parameters
    ----------
    frontmatter : dict[str, Any]
        Parsed YAML frontmatter from an ``orchid.md`` file.
    skip_sections : set[str] | None
        Sections to skip (e.g. ``{"storage"}`` for the CLI convention).

    Returns
    -------
    dict[str, str]
        Mapping of env-var name → value (strings).  The caller is
        responsible for applying them to ``os.environ``.
    """
    _skip = skip_sections or set()
    env_vars: dict[str, str] = {}

    for section, body in frontmatter.items():
        if not isinstance(body, dict):
            continue
        if section in _skip:
            continue
        for key, value in body.items():
            env_var = YAML_TO_ENV.get((section, key))
            if env_var is not None:
                env_vars[env_var] = str(value)

    logger.info(
        "[MD Config] Resolved %d infra env var(s) from frontmatter",
        len(env_vars),
    )
    return env_vars


def _resolve_agents_dir(
    root_path: Path,
    frontmatter: dict[str, Any],
    agents_dir: str | Path | None,
) -> Path:
    """Determine the agents directory from explicit arg, frontmatter, or default."""
    if agents_dir is not None:
        agents_dir_path = Path(agents_dir)
        if not agents_dir_path.is_absolute():
            agents_dir_path = root_path.parent / agents_dir_path
        return agents_dir_path

    agents_section = frontmatter.get("agents") if isinstance(frontmatter.get("agents"), dict) else {}
    dir_name = agents_section.get("agents_dir")
    if isinstance(dir_name, str):
        agents_dir_path = Path(dir_name)
        if not agents_dir_path.is_absolute():
            agents_dir_path = root_path.parent / agents_dir_path
        return agents_dir_path

    return root_path.parent / "agents"


def build_config_data_from_yaml(
    yaml_data: dict[str, Any],
    agent_configs: dict[str, dict[str, Any]],
    agent_behaviour_fields: frozenset[str],
) -> dict[str, Any]:
    """Build a validated config input dict from YAML data + assembled agents.

    Shared by ``load_md_config`` and ``_build_hybrid_config`` to avoid
    duplicating the section-filtering logic.

    Parameters
    ----------
    yaml_data : dict
        Parsed YAML frontmatter (MD) or ``yaml.safe_load`` output (YAML).
    agent_configs : dict
        Assembled per-agent config dicts keyed by agent name.
    agent_behaviour_fields : frozenset
        The set of field names on ``OrchidAgentsConfig`` — typically
        ``OrchidAgentsConfig.model_fields.keys()``.

    Returns
    -------
    dict[str, Any]
        Ready for ``OrchidAgentsConfig.model_validate``.
    """
    config_data: dict[str, Any] = {}

    for key, value in yaml_data.items():
        if key in agent_behaviour_fields:
            config_data[key] = value

    config_data["agents"] = agent_configs
    return config_data


def load_md_config(
    root_path: str | Path,
    agents_dir: str | Path | None = None,
) -> tuple[OrchidAgentsConfig, dict[str, str]]:
    """Load ``orchid.md`` + ``agents/*.md`` and produce an
    :class:`OrchidAgentsConfig`.

    Parameters
    ----------
    root_path : str | Path
        Path to the root ``orchid.md`` file.
    agents_dir : str | Path | None
        Directory containing per-agent ``.md`` files.  When ``None``,
        defaults to the ``agents_dir`` field in the root frontmatter
        (``agents.agents_dir``), then to ``<root_path.parent>/agents``.

    Returns
    -------
    tuple[OrchidAgentsConfig, dict[str, str]]
        - Fully-validated :class:`OrchidAgentsConfig` with defaults merged.
        - Mapping of ``file_path_string → sha256`` for all files read
          (root + agent MD files).  Used by the watcher for change detection.

    Raises
    ------
    FileNotFoundError
        If the root ``orchid.md`` file does not exist.
    ValueError
        If duplicate agent names are detected.
    pydantic.ValidationError
        If the assembled configuration does not match the schema.
    """
    root_path = Path(root_path)
    if not root_path.is_absolute():
        root_path = root_path.resolve()

    # ── 1. Load root MD ──────────────────────────────────────
    root_md = load_markdown_file(root_path)
    file_hashes: dict[str, str] = {str(root_path): root_md.sha256}
    root_fm = dict(root_md.frontmatter)

    # ── 2. Map infrastructure keys to env vars ────────────────
    # (not applied here — caller does that)

    # ── 3. Resolve agents directory ───────────────────────────
    agents_dir_path = _resolve_agents_dir(root_path, root_fm, agents_dir)

    # ── 4. Load agent MD files ────────────────────────────────
    agent_configs, agent_hashes = _load_agents(agents_dir_path)
    file_hashes.update(agent_hashes)

    # ── 5. Build OrchidAgentsConfig input dict ─────────────────
    config_data = build_config_data_from_yaml(root_fm, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

    # ── 6. Validate ───────────────────────────────────────────
    config = OrchidAgentsConfig.model_validate(config_data)

    agent_names = list(config.agents.keys())
    logger.info(
        "[MD Config] Loaded %d agent(s) from %s: %s",
        len(agent_names),
        root_path.name,
        agent_names,
    )

    return config, file_hashes
