"""Tests for the startup-hook signature validation in ``orchid_ai.bootstrap``."""

from __future__ import annotations

import pytest

from orchid_ai.bootstrap import _validate_startup_hook


class TestValidateStartupHook:
    def test_accepts_reader_and_settings(self):
        async def hook(reader, settings, **_):
            return None

        # Should not raise
        _validate_startup_hook("mod.hook", hook, {"reader": object(), "settings": None})

    def test_accepts_only_reader(self):
        async def hook(reader, **_):
            return None

        _validate_startup_hook("mod.hook", hook, {"reader": object(), "settings": None})

    def test_rejects_sync_function(self):
        def sync_hook(reader, settings, **_):
            return None

        with pytest.raises(TypeError, match="async function"):
            _validate_startup_hook("mod.hook", sync_hook, {"reader": object(), "settings": None})

    def test_rejects_mismatched_signature(self):
        async def hook(not_reader):
            return None

        with pytest.raises(TypeError, match="does not accept"):
            _validate_startup_hook("mod.hook", hook, {"reader": object(), "settings": None})

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            _validate_startup_hook("mod.hook", "not-a-function", {"reader": object(), "settings": None})

    def test_async_check_wins_over_signature_mismatch(self):
        """Async check must fire before signature bind_partial — a sync function
        with a mismatched signature should surface the `async` error (more
        informative) rather than the signature error."""

        def sync_with_bad_sig(wrong_name):
            return None

        with pytest.raises(TypeError, match="async function"):
            _validate_startup_hook("mod.hook", sync_with_bad_sig, {"reader": object(), "settings": None})
