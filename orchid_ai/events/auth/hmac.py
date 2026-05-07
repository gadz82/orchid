"""HMAC-SHA256 signature validator.

Compares the ``X-Orchid-Signature`` header (defaulting to
``sha256=<hex>``) against ``HMAC-SHA256(secret, raw_body)`` in
constant time.  Mirrors the GitHub / Stripe webhook pattern that
integrators expect.

Constructor parameters (from YAML ``validator.extra_args`` plus the
loader-resolved ``secret``):

- ``secret`` — the shared secret as a str (already decoded from
  ``env:VAR``).
- ``signature_header`` — header name (default ``"X-Orchid-Signature"``).
- ``algorithm`` — hash name passed to :func:`hmac.new` (default
  ``"sha256"``).
- ``prefix`` — optional ``"<algo>="`` prefix on the signature value
  (default ``"sha256="``).  Set to ``""`` to accept a bare hex digest.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac

from ...core.events.errors import SignalAuthValidationError
from .base import SignalAuthRequest, SignalAuthValidator


class HMACValidator(SignalAuthValidator):
    def __init__(
        self,
        *,
        secret: str,
        signature_header: str = "X-Orchid-Signature",
        algorithm: str = "sha256",
        prefix: str = "sha256=",
    ) -> None:
        if not secret:
            raise ValueError("HMACValidator requires a non-empty secret")
        self._secret = secret.encode("utf-8")
        self._header = signature_header.lower()
        self._algorithm = algorithm
        self._prefix = prefix

    async def validate(self, request: SignalAuthRequest) -> None:
        provided = request.headers.get(self._header)
        if not provided:
            raise SignalAuthValidationError(f"missing {self._header!r} header for source {request.source_id!r}")
        if self._prefix and not provided.startswith(self._prefix):
            raise SignalAuthValidationError(
                f"signature for source {request.source_id!r} missing {self._prefix!r} prefix"
            )
        provided_hex = provided[len(self._prefix) :] if self._prefix else provided

        try:
            mac = _hmac.new(
                self._secret,
                request.raw_body,
                getattr(hashlib, self._algorithm),
            )
        except AttributeError as exc:
            raise SignalAuthValidationError(f"HMAC algorithm {self._algorithm!r} is not supported by hashlib") from exc

        expected_hex = mac.hexdigest()
        # Constant-time comparison so an attacker probing differential
        # response timing learns nothing about which prefix matched.
        if not _hmac.compare_digest(expected_hex, provided_hex.lower()):
            raise SignalAuthValidationError(f"signature mismatch for source {request.source_id!r}")
