"""Tests for general-purpose utilities."""

from __future__ import annotations

import pytest

from orchid_ai.utils import import_class, unwrap_exception_group


class TestUnwrapExceptionGroup:
    """``unwrap_exception_group`` flattens TaskGroup/ExceptionGroup summaries."""

    def test_returns_non_group_unchanged(self):
        exc = ValueError("plain error")
        assert unwrap_exception_group(exc) is exc

    def test_unwraps_single_exception_group(self):
        inner = RuntimeError("real failure")
        group = ExceptionGroup("outer", [inner])
        assert unwrap_exception_group(group) is inner

    def test_unwraps_nested_exception_group(self):
        inner = ConnectionError("network down")
        nested = ExceptionGroup("nested", [inner])
        group = ExceptionGroup("outer", [nested])
        assert unwrap_exception_group(group) is inner


class TestImportClass:
    """``import_class`` resolves dotted class paths."""

    def test_imports_builtin(self):
        cls = import_class("builtins.ValueError")
        assert cls is ValueError

    def test_raises_on_missing_class(self):
        with pytest.raises(ImportError):
            import_class("builtins.DoesNotExist")
