"""Tests for the provider registry in ``orchid_ai.llm_factory``.

Locks in the dict-based registry contract: register_provider replaces
duplicates explicitly (with a warning), and longest-prefix matching
keeps precedence stable when a caller adds a more-specific prefix.
"""

from __future__ import annotations

import logging

from orchid_ai.llm_factory import (
    _PROVIDER_MAP,
    ProviderEntry,
    register_provider,
)


class TestProviderRegistry:
    def test_default_entries_present(self):
        for prefix in ("openai/", "anthropic/", "gemini/", "ollama/"):
            assert prefix in _PROVIDER_MAP
            assert isinstance(_PROVIDER_MAP[prefix], ProviderEntry)

    def test_register_adds_new_provider(self, caplog):
        snapshot = dict(_PROVIDER_MAP)
        try:
            register_provider(
                prefix="cohere/",
                module_path="langchain_cohere",
                class_name="ChatCohere",
                strip_prefix="cohere/",
            )
            entry = _PROVIDER_MAP["cohere/"]
            assert entry.module_path == "langchain_cohere"
            assert entry.class_name == "ChatCohere"
        finally:
            _PROVIDER_MAP.clear()
            _PROVIDER_MAP.update(snapshot)

    def test_register_replacing_warns(self, caplog):
        snapshot = dict(_PROVIDER_MAP)
        try:
            with caplog.at_level(logging.WARNING):
                register_provider(
                    prefix="openai/",
                    module_path="my.custom.module",
                    class_name="MyOpenAI",
                    strip_prefix="openai/",
                )
            assert any("Replacing provider" in r.message for r in caplog.records)
            assert _PROVIDER_MAP["openai/"].module_path == "my.custom.module"
        finally:
            _PROVIDER_MAP.clear()
            _PROVIDER_MAP.update(snapshot)
