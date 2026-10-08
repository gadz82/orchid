"""Plugin-package hints for dotted-class-path import errors.

Storage backends that live in separate plugin packages are referenced by
dotted class path.  When such a path cannot be imported the factories
append an actionable ``pip install`` hint using the prefix map below.
"""

from __future__ import annotations

#: Known plugin module prefixes → the PyPI package that provides them.
PLUGIN_PACKAGE_HINTS: dict[str, str] = {
    "orchid_storage_postgres.": "orchid-storage-postgres",
    "orchid_storage_sqlite.": "orchid-storage-sqlite",
}


def plugin_hint(class_path: str) -> str | None:
    """Return the package that provides ``class_path``, or ``None`` when unknown."""
    for prefix, package in PLUGIN_PACKAGE_HINTS.items():
        if class_path.startswith(prefix):
            return package
    return None


def with_plugin_hint(message: str, class_path: str) -> str:
    """Append the ``pip install <package>`` hint for known plugin prefixes."""
    package = plugin_hint(class_path)
    if package is None:
        return message
    return f"{message} Install the missing plugin: pip install {package}"


__all__ = ["PLUGIN_PACKAGE_HINTS", "plugin_hint", "with_plugin_hint"]
