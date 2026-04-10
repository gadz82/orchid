"""Date formatting built-in tools."""

from __future__ import annotations

from datetime import datetime


def format_date(value: str, fmt: str = "%Y-%m-%d") -> str:
    """
    Parse a date string and re-format it.

    Tries ISO-8601 first, then common variants.

    Parameters
    ----------
    value : str
        Input date string (e.g. ``"2025-03-15T10:30:00Z"``).
    fmt : str
        Output format (``strftime`` pattern).  Default: ``"%Y-%m-%d"``.

    Returns
    -------
    str
        Formatted date string.
    """
    formats_to_try = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    for parse_fmt in formats_to_try:
        try:
            dt = datetime.strptime(value, parse_fmt)
            return dt.strftime(fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date '{value}' with any known format")
