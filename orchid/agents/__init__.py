"""
Agent implementations.

The library provides ``GenericAgent`` (config-driven, no subclassing needed).
Custom agents live outside ``src/`` (e.g. ``examples/helpdesk/agents/``) and are
resolved at runtime via dotted import paths in ``agents.yaml``.
"""
