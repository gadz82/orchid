"""
Built-in tools — in-process Python functions (ADR-017).

Bundled utilities shipped with the framework:
  - ``orchid.tools.dates.format_date`` — parse + reformat dates
  - ``orchid.tools.math.calculate_completion_rate`` — simple percentage

These are generic helpers, not domain-specific.  Domain-specific tools
belong in consumer projects (e.g. ``docebo.tools``, ``examples.*.tools``).
"""
