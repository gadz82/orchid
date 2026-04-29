"""Performance logger configuration.

The framework emits ``[PERF]`` lines from a dedicated ``orchid.perf``
logger so operators can profile request flow without changing source
code.  These lines are noisy — useful when investigating a slow
request, distracting the rest of the time — so they are **off by
default**.

Enable them either by:

- Setting ``ORCHID_ENABLE_PERF_LOGS=true`` in the process environment
  (read once at import time and on every explicit
  :func:`configure_perf_logger` call with ``enabled=None``).
- Calling :func:`configure_perf_logger(True)` programmatically before
  the first request.

The toggle works by setting the logger's effective level: WARNING
silences the ``info()`` calls scattered across the framework; INFO
lets them through to whatever handlers the application has wired up.
"""

from __future__ import annotations

import logging
import os

PERF_LOGGER_NAME = "orchid.perf"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

_ENV_VAR = "ORCHID_ENABLE_PERF_LOGS"


def _read_env_flag() -> bool:
    return os.environ.get(_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def configure_perf_logger(enabled: bool | None = None) -> bool:
    """Configure the ``orchid.perf`` logger level.

    Parameters
    ----------
    enabled : bool | None
        - ``True``: emit perf logs at INFO.
        - ``False``: silence perf logs (level WARNING).
        - ``None`` (default): read from ``ORCHID_ENABLE_PERF_LOGS`` env var.

    Returns
    -------
    bool
        The resolved enabled state, useful when callers want to log the
        outcome (e.g. "[API] Perf logs enabled via env var").
    """
    if enabled is None:
        enabled = _read_env_flag()
    logger = logging.getLogger(PERF_LOGGER_NAME)
    logger.setLevel(logging.INFO if enabled else logging.WARNING)
    logger.propagate = True
    return enabled


# Configure once at import time so embedded callers (no explicit
# bootstrap) still get the env-var-driven default.
configure_perf_logger()
