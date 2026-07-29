"""
YAML config loader with environment variable interpolation.

Syntax: ``${VAR_NAME}`` in any YAML string value is replaced with
``os.environ["VAR_NAME"]``.  Missing variables raise ``ValueError``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml

from .errors import OrchidConfigError
from .schema import OrchidAgentsConfig

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)}")


def _interpolate_env(raw: str) -> str:
    """Replace ``${VAR}`` placeholders with environment variable values.

    Only processes non-comment portions of each line.  YAML comments
    (everything after ``#``) are left untouched.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
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

    Parameters
    ----------
    path : str | Path
        Path to the YAML file (absolute or relative to cwd).

    Returns
    -------
    OrchidAgentsConfig
        Fully validated configuration with defaults merged in.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If a ``${VAR}`` references an unset environment variable.
    pydantic.ValidationError
        If the YAML content does not match the schema.
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
