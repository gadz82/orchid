"""Calculation built-in tools."""

from __future__ import annotations


def calculate_completion_rate(enrolled: int, completed: int) -> float:
    """
    Calculate course completion percentage.

    Parameters
    ----------
    enrolled : int
        Total number of enrolled users.
    completed : int
        Number of users who completed the course.

    Returns
    -------
    float
        Completion rate as a percentage (0.0–100.0), rounded to 1 decimal.
    """
    if enrolled <= 0:
        return 0.0
    return round((completed / enrolled) * 100, 1)
