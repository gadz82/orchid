from __future__ import annotations

from orchid.tools.math import calculate_completion_rate


def test_normal_case():
    assert calculate_completion_rate(100, 75) == 75.0


def test_zero_enrolled():
    assert calculate_completion_rate(0, 10) == 0.0


def test_negative_enrolled():
    assert calculate_completion_rate(-5, 3) == 0.0


def test_all_completed():
    assert calculate_completion_rate(50, 50) == 100.0


def test_none_completed():
    assert calculate_completion_rate(50, 0) == 0.0


def test_rounding():
    assert calculate_completion_rate(3, 1) == 33.3
