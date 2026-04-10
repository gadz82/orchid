from __future__ import annotations

import pytest

from orchid.tools.dates import format_date


def test_iso8601_with_z_suffix():
    assert format_date("2025-03-15T10:30:00Z") == "2025-03-15"


def test_iso8601_without_z():
    assert format_date("2025-03-15T10:30:00") == "2025-03-15"


def test_iso8601_with_timezone_offset():
    assert format_date("2025-03-15T10:30:00+02:00") == "2025-03-15"


def test_iso_with_space_separator():
    assert format_date("2025-03-15 10:30:00") == "2025-03-15"


def test_date_only():
    assert format_date("2025-03-15") == "2025-03-15"


def test_dd_mm_yyyy_format():
    assert format_date("15/03/2025") == "2025-03-15"


def test_mm_dd_yyyy_format():
    # 01/15/2025 cannot match DD/MM (month=15 is invalid), so it falls through to MM/DD
    assert format_date("01/15/2025") == "2025-01-15"


def test_custom_output_format():
    assert format_date("2025-03-15T10:30:00Z", fmt="%d %B %Y") == "15 March 2025"


def test_invalid_date_raises_value_error():
    with pytest.raises(ValueError, match="Could not parse date"):
        format_date("not-a-date")
