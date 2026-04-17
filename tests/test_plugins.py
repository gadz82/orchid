"""Tests for ``orchid_ai.plugins.iter_entry_point_plugins``."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from orchid_ai.plugins import iter_entry_point_plugins


def _fake_ep(name: str, load_result=None, load_exc: Exception | None = None) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    if load_exc is not None:
        ep.load.side_effect = load_exc
    else:
        ep.load.return_value = load_result
    return ep


def _mock_entry_points(group: str, eps: list) -> MagicMock:
    """Build a MagicMock that mimics either ``eps.select(group=...)`` or ``eps.get(group, [])``."""
    result = MagicMock()
    result.select.return_value = eps  # 3.12+
    result.get.return_value = eps  # 3.9-3.11
    return result


class TestIterEntryPointPlugins:
    def test_yields_loaded_plugins(self):
        plugin_obj = object()
        ep = _fake_ep("my_plugin", load_result=plugin_obj)

        with patch("orchid_ai.plugins.entry_points", return_value=_mock_entry_points("foo", [ep]), create=True):
            pass  # the real import path is import-local; patch the symbol that gets resolved.

        # The helper uses ``from importlib.metadata import entry_points`` inside
        # the function, so patch that module attribute directly.
        with patch("importlib.metadata.entry_points", return_value=_mock_entry_points("foo", [ep])):
            result = list(iter_entry_point_plugins("foo"))

        assert result == [("my_plugin", plugin_obj)]

    def test_skips_failing_plugin_and_logs(self, caplog):
        good = _fake_ep("good", load_result="ok")
        bad = _fake_ep("bad", load_exc=RuntimeError("boom"))

        with patch("importlib.metadata.entry_points", return_value=_mock_entry_points("x", [good, bad])):
            with caplog.at_level(logging.WARNING):
                result = list(iter_entry_point_plugins("x"))

        assert result == [("good", "ok")]
        assert any("Failed to load 'bad'" in r.message for r in caplog.records)

    def test_empty_group_yields_nothing(self):
        with patch("importlib.metadata.entry_points", return_value=_mock_entry_points("none", [])):
            assert list(iter_entry_point_plugins("none")) == []

    def test_uses_provided_logger(self, caplog):
        custom = logging.getLogger("orchid_ai.plugins.test-custom")
        bad = _fake_ep("bad", load_exc=ValueError("nope"))

        with patch("importlib.metadata.entry_points", return_value=_mock_entry_points("g", [bad])):
            with caplog.at_level(logging.WARNING, logger="orchid_ai.plugins.test-custom"):
                list(iter_entry_point_plugins("g", logger=custom))

        assert any(r.name == "orchid_ai.plugins.test-custom" for r in caplog.records)


@pytest.mark.parametrize("has_select", [True, False])
def test_compat_select_vs_get(has_select):
    """Python 3.12+ uses ``eps.select(group=...)``; older uses ``eps.get(group, [])``."""
    plugin = _fake_ep("p", load_result=1)

    class FakeEPs:
        pass

    eps = FakeEPs()
    if has_select:
        eps.select = MagicMock(return_value=[plugin])
    else:
        eps.get = MagicMock(return_value=[plugin])

    with patch("importlib.metadata.entry_points", return_value=eps):
        result = list(iter_entry_point_plugins("g"))

    assert result == [("p", 1)]
    if has_select:
        eps.select.assert_called_once_with(group="g")
    else:
        eps.get.assert_called_once_with("g", [])
