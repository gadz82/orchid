"""Static-bearer validator.

For sources whose only requirement is "did the caller present THIS
secret bearer token?".  Compares the ``Authorization: Bearer <token>``
header against the configured token in constant time.

Useful for:

- Internal service calls where mTLS is overkill.
- Trusted webhook senders that don't sign payloads (rare; prefer
  HMAC when possible).

Constructor parameters:

- ``secret`` — the expected token (already resolved from
  ``env:VAR``).
- ``header`` — header name (default ``"Authorization"``).
- ``scheme`` — auth scheme (default ``"Bearer"``).  Set to ``""``
  to accept the raw token.
"""

from __future__ import annotations

import hmac as _hmac

from ...core.events.errors import SignalAuthValidationError
from .base import SignalAuthRequest, SignalAuthValidator


class BearerValidator(SignalAuthValidator):
    def __init__(
        self,
        *,
        secret: str,
        header: str = "Authorization",
        scheme: str = "Bearer",
    ) -> None:
        if not secret:
            raise ValueError("BearerValidator requires a non-empty secret")
        self._secret = secret
        self._header = header.lower()
        self._scheme = scheme

    async def validate(self, request: SignalAuthRequest) -> None:
        raw = request.headers.get(self._header, "")
        if self._scheme:
            prefix = f"{self._scheme} "
            if not raw.startswith(prefix):
                raise SignalAuthValidationError(
                    f"missing {self._scheme!r} authentication for source {request.source_id!r}"
                )
            provided = raw[len(prefix) :]
        else:
            provided = raw

        if not provided:
            raise SignalAuthValidationError(f"empty bearer token for source {request.source_id!r}")
        if not _hmac.compare_digest(self._secret, provided):
            raise SignalAuthValidationError(f"bearer mismatch for source {request.source_id!r}")
