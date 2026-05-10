"""Tests for ``HMACValidator`` and ``BearerValidator``.

These run against the validator ABC alone — no FastAPI / network
involvement.  The HTTP ingestion path's tests live in
``test_http_ingestion.py``.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac

import pytest

from orchid_ai.core.events.errors import SignalAuthValidationError
from orchid_ai.events.auth import BearerValidator, HMACValidator
from orchid_ai.events.auth.base import SignalAuthRequest


# ── HMAC ────────────────────────────────────────────────────


def _signed(secret: str, body: bytes, *, prefix: str = "sha256=") -> str:
    return prefix + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def test_hmac_validates_correct_signature() -> None:
    body = b'{"type":"x","tenant_key":"t-1"}'
    v = HMACValidator(secret="topsecret")
    await v.validate(
        SignalAuthRequest(
            source_id="src",
            raw_body=body,
            headers={"x-orchid-signature": _signed("topsecret", body)},
        )
    )


async def test_hmac_rejects_missing_header() -> None:
    v = HMACValidator(secret="s")
    with pytest.raises(SignalAuthValidationError, match="missing"):
        await v.validate(SignalAuthRequest(source_id="src", raw_body=b"x", headers={}))


async def test_hmac_rejects_missing_prefix() -> None:
    v = HMACValidator(secret="s")
    with pytest.raises(SignalAuthValidationError, match="prefix"):
        await v.validate(
            SignalAuthRequest(
                source_id="src",
                raw_body=b"x",
                headers={"x-orchid-signature": "deadbeef"},
            )
        )


async def test_hmac_rejects_wrong_signature() -> None:
    body = b"hello"
    v = HMACValidator(secret="topsecret")
    bad = _signed("WRONG", body)
    with pytest.raises(SignalAuthValidationError, match="mismatch"):
        await v.validate(
            SignalAuthRequest(
                source_id="src",
                raw_body=body,
                headers={"x-orchid-signature": bad},
            )
        )


async def test_hmac_no_prefix_mode_accepts_bare_hex() -> None:
    body = b"x"
    v = HMACValidator(secret="s", prefix="")
    bare = _hmac.new(b"s", body, hashlib.sha256).hexdigest()
    await v.validate(
        SignalAuthRequest(
            source_id="src",
            raw_body=body,
            headers={"x-orchid-signature": bare},
        )
    )


async def test_hmac_rejects_unknown_algorithm() -> None:
    v = HMACValidator(secret="s", algorithm="not-an-algo")
    body = b"x"
    with pytest.raises(SignalAuthValidationError, match="algorithm"):
        await v.validate(
            SignalAuthRequest(
                source_id="src",
                raw_body=body,
                headers={"x-orchid-signature": "sha256=deadbeef"},
            )
        )


def test_hmac_rejects_empty_secret() -> None:
    with pytest.raises(ValueError):
        HMACValidator(secret="")


# ── Bearer ──────────────────────────────────────────────────


async def test_bearer_validates_correct_token() -> None:
    v = BearerValidator(secret="topsecret")
    await v.validate(
        SignalAuthRequest(
            source_id="src",
            raw_body=b"",
            headers={"authorization": "Bearer topsecret"},
        )
    )


async def test_bearer_rejects_wrong_token() -> None:
    v = BearerValidator(secret="topsecret")
    with pytest.raises(SignalAuthValidationError, match="mismatch"):
        await v.validate(
            SignalAuthRequest(
                source_id="src",
                raw_body=b"",
                headers={"authorization": "Bearer WRONG"},
            )
        )


async def test_bearer_rejects_missing_scheme() -> None:
    v = BearerValidator(secret="s")
    with pytest.raises(SignalAuthValidationError, match="Bearer"):
        await v.validate(
            SignalAuthRequest(
                source_id="src",
                raw_body=b"",
                headers={"authorization": "Token s"},
            )
        )


async def test_bearer_no_scheme_mode_accepts_raw() -> None:
    v = BearerValidator(secret="raw", scheme="")
    await v.validate(
        SignalAuthRequest(
            source_id="src",
            raw_body=b"",
            headers={"authorization": "raw"},
        )
    )


def test_bearer_rejects_empty_secret() -> None:
    with pytest.raises(ValueError):
        BearerValidator(secret="")
