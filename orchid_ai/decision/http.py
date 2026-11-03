"""Shared HTTP transport for decision providers.

Every adapter posts JSON through :class:`DecisionHttpClient`. Keeping transport,
retries, breaker state, size caps, and error mapping in one place means adapters
stay thin and a provider outage behaves the same at every decision point.

The rules here are deliberately conservative: HTTPS only, an explicit host
allowlist, no redirects, no environment proxies, bounded response bodies, and
errors that never carry the request or response body.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..config.schema_decision import normalise_host, validate_decision_base_url
from ..core.decision import OrchidDecisionError

__all__ = ["DecisionBreaker", "DecisionHttpClient"]

_MAX_BACKOFF_SECONDS = 2.0
_MAX_RETRY_AFTER_SECONDS = 5.0


class DecisionBreaker:
    """Per-provider circuit breaker.

    Consecutive failures open the breaker for ``cooldown_s``; after the cooldown
    one half-open probe is allowed. A success closes it, a failure re-opens it.
    State is synchronous, so no lock is needed in a single event loop.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._threshold = max(1, failure_threshold)
        self._cooldown_s = max(0.0, cooldown_s)
        self._clock = clock or (lambda: asyncio.get_running_loop().time())
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_inflight = False

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return self._clock() - self._opened_at < self._cooldown_s

    def allow(self) -> bool:
        """Return whether a call may proceed, reserving the half-open probe."""
        if self._opened_at is None:
            return True
        elapsed = self._clock() - self._opened_at
        if elapsed < self._cooldown_s:
            return False
        if self._half_open_inflight:
            return False
        self._half_open_inflight = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open_inflight = False

    def record_failure(self) -> None:
        self._failures += 1
        self._half_open_inflight = False
        if self._failures >= self._threshold:
            self._opened_at = self._clock()


class DecisionHttpClient:
    """Bounded JSON POST client for one decision provider.

    Parameters
    ----------
    base_url:
        HTTPS endpoint. Validated with the same rules as configuration.
    allowed_hosts:
        Normalised hostnames permitted to receive decision state.
    provider:
        Provider key, used in error metadata.
    api_key:
        Optional bearer token. The factory resolves it from the configured
        environment variable; it is never logged.
    max_retries:
        Retries after the first attempt for timeouts, 429, and 5xx.
    max_response_bytes:
        Hard upper bound on the response body.
    client:
        Injected ``httpx.AsyncClient`` (tests use ``MockTransport``). When
        omitted the client creates and owns one lazily.
    sleep / clock:
        Injected for deterministic retry and breaker tests.
    """

    def __init__(
        self,
        *,
        base_url: str,
        allowed_hosts: list[str],
        provider: str = "",
        api_key: str = "",
        timeout: float = 2.0,
        max_retries: int = 1,
        max_response_bytes: int = 262_144,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._provider = provider
        self._max_response_bytes = max(1024, int(max_response_bytes))
        self._max_retries = max(0, int(max_retries))
        self._timeout = max(0.1, float(timeout))
        self._api_key = api_key
        self._sleep = sleep or asyncio.sleep
        try:
            self._base_url = validate_decision_base_url(base_url, provider=provider or "decision")
        except ValueError as exc:
            raise OrchidDecisionError(str(exc), code="egress_denied", provider=provider, retryable=False) from exc

        self._host = normalise_host(urlsplit(self._base_url).hostname or "", label=f"provider {provider!r}")
        normalised_allowed = {
            normalise_host(entry, label="allowed_decision_hosts") for entry in allowed_hosts if entry.strip()
        }
        if self._host not in normalised_allowed:
            raise OrchidDecisionError(
                f"provider host {self._host!r} is not in allowed_decision_hosts",
                code="egress_denied",
                provider=provider,
                retryable=False,
                host=self._host,
            )

        self._client = client
        self._owns_client = client is None
        self._breaker = DecisionBreaker(
            failure_threshold=failure_threshold,
            cooldown_s=cooldown_s,
            clock=clock,
        )

    @property
    def host(self) -> str:
        return self._host

    async def request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` and return the decoded JSON object.

        Raises :class:`OrchidDecisionError` with a stable code. The message,
        status, and host are safe to log; no body or state is included.
        """
        if not self._breaker.allow():
            raise OrchidDecisionError(
                "provider breaker is open",
                code="provider_unavailable",
                provider=self._provider,
                retryable=True,
                host=self._host,
            )

        attempts = self._max_retries + 1
        last_error: OrchidDecisionError | None = None

        for attempt in range(1, attempts + 1):
            try:
                status, headers, body = await self._send(path, payload)
            except OrchidDecisionError as exc:
                last_error = exc
                self._breaker.record_failure()
                if not exc.retryable or attempt >= attempts:
                    raise
                await self._sleep(self._backoff(attempt, None))
                continue

            if status == 429 or 500 <= status < 600:
                self._breaker.record_failure()
                last_error = OrchidDecisionError(
                    f"provider returned HTTP {status}",
                    code="provider_unavailable",
                    provider=self._provider,
                    retryable=True,
                    status_code=status,
                    host=self._host,
                )
                if attempt >= attempts:
                    raise last_error
                await self._sleep(self._backoff(attempt, headers.get("retry-after")))
                continue

            if status >= 400:
                self._breaker.record_failure()
                raise OrchidDecisionError(
                    f"provider returned HTTP {status}",
                    code="provider_unavailable",
                    provider=self._provider,
                    retryable=False,
                    status_code=status,
                    host=self._host,
                )

            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._breaker.record_failure()
                raise OrchidDecisionError(
                    "provider returned invalid JSON",
                    code="invalid_response",
                    provider=self._provider,
                    retryable=False,
                    host=self._host,
                ) from exc
            if not isinstance(data, dict):
                self._breaker.record_failure()
                raise OrchidDecisionError(
                    "provider response must be a JSON object",
                    code="invalid_response",
                    provider=self._provider,
                    retryable=False,
                    host=self._host,
                )
            self._breaker.record_success()
            return data

        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    # ── Internals ────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                trust_env=False,
            )
            self._owns_client = True
        return self._client

    def _url(self, path: str) -> str:
        return f"{self._base_url.rstrip('/')}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _send(self, path: str, payload: dict[str, Any]) -> tuple[int, httpx.Headers, bytes]:
        client = self._get_client()
        try:
            async with client.stream("POST", self._url(path), json=payload, headers=self._headers()) as response:
                status = response.status_code
                headers = response.headers
                if status >= 400:
                    return status, headers, b""
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_response_bytes:
                        raise OrchidDecisionError(
                            "provider response exceeded the size limit",
                            code="invalid_response",
                            provider=self._provider,
                            retryable=False,
                            status_code=status,
                            host=self._host,
                        )
                    chunks.append(chunk)
                return status, headers, b"".join(chunks)
        except OrchidDecisionError:
            raise
        except httpx.TimeoutException as exc:
            raise OrchidDecisionError(
                "provider timed out",
                code="timeout",
                provider=self._provider,
                retryable=True,
                host=self._host,
            ) from exc
        except httpx.TransportError as exc:
            raise OrchidDecisionError(
                "provider transport failed",
                code="provider_unavailable",
                provider=self._provider,
                retryable=True,
                host=self._host,
            ) from exc

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            seconds = _parse_retry_after(retry_after)
            if seconds is not None:
                return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)
        return min(0.25 * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)


def _parse_retry_after(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
