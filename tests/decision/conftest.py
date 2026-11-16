"""Shared fixtures for decision-adapter tests.

Every test in this folder injects an ``httpx.MockTransport``. The autouse guard
makes an accidental real ``AsyncClient`` fail loudly instead of reaching the
network.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    original = httpx.AsyncClient.__init__

    def guarded(self, *args, **kwargs):
        assert kwargs.get("transport") is not None, (
            "decision tests must inject an httpx.MockTransport; a real network client was constructed"
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", guarded)
