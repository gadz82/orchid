"""
YAML config loader with environment variable interpolation.

Syntax:
  * ``${VAR_NAME}`` is replaced with ``os.environ["VAR_NAME"]``.
  * ``${VAR_NAME:-default}`` uses ``default`` when the variable is unset.

Missing variables without a default raise :class:`OrchidConfigError`.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .errors import OrchidConfigError
from .schema import OrchidAgentsConfig
from .schema_agent import _deep_merge

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?}")


def _interpolate_env(raw: str) -> str:
    """Replace ``${VAR_NAME}`` placeholders with environment variable values.

    Supports an optional default: ``${VAR_NAME:-default}`` is replaced
    with the environment variable value when set, otherwise ``default``.
    When no default is provided and the variable is unset, an
    :class:`OrchidConfigError` is raised.

    Only processes non-comment portions of each line.  YAML comments
    (everything after ``#``) are left untouched.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        value = os.environ.get(var_name)
        if value is None:
            if default is not None:
                return default
            raise OrchidConfigError(
                f"Environment variable '{var_name}' is referenced in agents.yaml "
                f"but not set. Add it to your .env or environment."
            )
        return value

    lines = []
    for line in raw.splitlines(keepends=True):
        # Split at the first '#' that is NOT inside a YAML value
        # Simple heuristic: if '#' appears, only interpolate the part before it
        comment_idx = _find_comment_start(line)
        if comment_idx is not None:
            code_part = line[:comment_idx]
            comment_part = line[comment_idx:]
            lines.append(_ENV_VAR_RE.sub(_replace, code_part) + comment_part)
        else:
            lines.append(_ENV_VAR_RE.sub(_replace, line))
    return "".join(lines)


def _find_comment_start(line: str) -> int | None:
    """Find the index of the first ``#`` that starts a YAML comment.

    Skips ``#`` characters inside quoted strings.
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return i
    return None


def load_config(path: str | Path) -> OrchidAgentsConfig:
    """
    Load and validate the agents YAML configuration.

    ``path`` may be either a single YAML file or a directory containing
    multiple ``*.yaml`` / ``*.yml`` files.  When a directory is given,
    every YAML file is loaded and deep-merged into a single config.
    The ``agents`` dictionary is merged across files; defining the same
    agent name in more than one file raises :class:`OrchidConfigError`.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file or directory (absolute or relative to cwd).

    Returns
    -------
    OrchidAgentsConfig
        Fully validated configuration with defaults merged in.

    Raises
    ------
    FileNotFoundError
        If the YAML file or directory does not exist.
    ValueError
        If a ``${VAR}`` references an unset environment variable.
    pydantic.ValidationError
        If the YAML content does not match the schema.
    OrchidConfigError
        If the same agent name is declared in multiple files.
    """
    path = Path(path)
    if not path.is_absolute():
        # Try relative to this package's parent (agents/)
        agents_dir = Path(__file__).resolve().parent.parent.parent
        candidate = agents_dir / path
        if candidate.exists():
            path = candidate
        # Otherwise use cwd-relative (let it fail naturally if not found)

    if not path.exists():
        raise FileNotFoundError(f"Agents config not found: {path}")

    if path.is_dir():
        return load_config_directory(path)

    raw_text = path.read_text(encoding="utf-8")

    # Environment variable interpolation before YAML parsing
    interpolated = _interpolate_env(raw_text)

    data = yaml.safe_load(interpolated)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML dict at top level, got {type(data).__name__}")

    config = OrchidAgentsConfig.model_validate(data)

    agent_names = list(config.agents.keys())
    logger.info("[Config] Loaded %d agents from %s: %s", len(agent_names), path.name, agent_names)

    return config


def load_config_directory(path: Path) -> OrchidAgentsConfig:
    """Load and merge every ``*.yaml`` / ``*.yml`` file in a directory.

    Files are processed in deterministic alphabetical order.  Top-level
    dictionaries are deep-merged; the ``agents`` map is merged across
    files but duplicate agent names are rejected.
    """
    files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
    if not files:
        raise OrchidConfigError(f"No YAML files found in config directory: {path}")

    merged: dict[str, Any] = {}
    seen_agents: dict[str, str] = {}

    for file in files:
        raw_text = file.read_text(encoding="utf-8")
        interpolated = _interpolate_env(raw_text)
        data = yaml.safe_load(interpolated)
        if not isinstance(data, dict):
            raise TypeError(f"Expected YAML dict in {file}, got {type(data).__name__}")

        file_agents = data.get("agents")
        if isinstance(file_agents, dict):
            for agent_name in file_agents:
                if agent_name in seen_agents:
                    raise OrchidConfigError(
                        f"Agent '{agent_name}' is defined in both {seen_agents[agent_name]} and {file.name}"
                    )
                seen_agents[agent_name] = file.name

        merged = _deep_merge(merged, data)

    config = OrchidAgentsConfig.model_validate(merged)

    agent_names = list(config.agents.keys())
    logger.info(
        "[Config] Loaded %d agents from directory %s: %s",
        len(agent_names),
        path.name,
        agent_names,
    )

    return config
