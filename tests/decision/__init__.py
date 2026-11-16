"""Test package marker.

Pytest's default ``prepend`` import mode derives module names from the file
basename. This marker keeps ``tests/decision/test_registry.py`` distinct from
``tests/tools/test_registry.py`` when the full suite is collected.
"""
