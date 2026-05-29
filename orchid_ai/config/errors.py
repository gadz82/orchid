"""Configuration-related error types."""

from __future__ import annotations


class OrchidConfigError(ValueError):
    """Raised when configuration loading or interpolation fails.

    Subclasses ``ValueError`` for backward compatibility — existing
    ``except ValueError`` handlers continue to work.  New code should
    catch ``OrchidConfigError`` directly for precision::

        try:
            config = load_config("agents.yaml")
        except OrchidConfigError as exc:
            logger.error("Config invalid: %s", exc)
            sys.exit(1)
    """
