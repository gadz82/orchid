"""Shared HTTP transport for decision providers.

Every adapter posts JSON through :class:`DecisionHttpClient`. Keeping transport,
retries, breaker state, size caps, and error mapping in one place means adapters
stay thin and a provider outage behaves the same at every decision point.

The rules here are deliberately conservative: HTTPS only, an explicit host
allowlist, resolved-address checks before every call, no redirects, no
environment proxies, a whole-call deadline, bounded response bodies, and errors
that never carry the request or response body.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import random
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..config.schema_decision import normalise_host, validate_decision_base_url, validate_resolved_address
from ..core.decision import OrchidDecisionError

__all__ = ["DecisionBreaker", "DecisionHttpClient"]

_MAX_BACKOFF_SECONDS = 2.0
_MAX_RETRY_AFTER_SECONDS = 5.0
_MAX_RESPONSE_BYTES = 1_048_576
_MIN_RESPONSE_BYTES = 1024


async def _default_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


class DecisionBreaker:
    """Per-provider circuit breaker.

    Consecutive failed *calls* open the breaker for ``cooldown_s``; after the
    cooldown one half-open probe is allowed. A success closes it, a failure
    re-opens it. ``abandon`` releases the probe reservation when a call is
    cancelled, so a budget timeout cannot wedge the provider forever.
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

    def abandon(self) -> None:
        """Release a half-open reservation without recording an outcome."""
        self._half_open_inflight = False

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
    timeout:
        Whole-call budget in seconds, including retries and backoff. It is not
        a per-read timeout: a slow-drip provider cannot hold a turn open.
    max_retries:
        Retries after the first attempt for timeouts, 429, and 5xx.
    max_response_bytes:
        Hard upper bound on the decoded response body.
    resolver:
        Injected host resolver for tests. The default uses the event loop's
        ``getaddrinfo``. Every returned address is checked against the egress
        policy before the request starts.
    client:
        Injected ``httpx.AsyncClient`` (tests use ``MockTransport``). It must
        disable redirects. When omitted the client creates and owns one lazily.
    sleep / clock / jitter:
        Injected for deterministic retry and breaker tests.
    """

    # ORCHID-MEM: 01a1a14c-6b7e-7f5a-9a3d-3f5b7c9e1d20 — Bounded decision transport with egress, deadline, and breaker.

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
        resolver: Callable[[str], Awaitable[list[str]] | list[str]] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        self._provider = provider
        self._max_response_bytes = min(_MAX_RESPONSE_BYTES, max(_MIN_RESPONSE_BYTES, int(max_response_bytes)))
        self._max_retries = max(0, int(max_retries))
        self._timeout = max(0.1, float(timeout))
        self._api_key = api_key
        self._resolver = resolver or _default_resolver
        self._sleep = sleep or asyncio.sleep
        self._jitter = jitter or random.random
        self._closed = False
        try:
            self._base_url = validate_decision_base_url(base_url, provider=provider or "decision")
        except ValueError as exc:
            raise OrchidDecisionError(str(exc), code="egress_denied", provider=provider, retryable=False) from None

        self._host = normalise_host(urlsplit(self._base_url).hostname or "", label=f"provider {provider!r}")
        try:
            normalised_allowed = {
                normalise_host(entry, label="allowed_decision_hosts") for entry in allowed_hosts if entry.strip()
            }
        except ValueError as exc:
            raise OrchidDecisionError(str(exc), code="egress_denied", provider=provider, retryable=False) from None
        if self._host not in normalised_allowed:
            raise OrchidDecisionError(
                f"provider host {self._host!r} is not in allowed_decision_hosts",
                code="egress_denied",
                provider=provider,
                retryable=False,
                host=self._host,
            )

        if client is not None and getattr(client, "follow_redirects", False):
            raise OrchidDecisionError(
                "injected decision client must disable redirects",
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

    async def request_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST ``payload`` and return the decoded JSON object.

        ``timeout`` overrides the client budget for this call; the turn budget
        can pass its remaining time here. Raises :class:`OrchidDecisionError`
        with a stable code. The message, status, and host are safe to log; no
        body or state is included.
        """
        if self._closed:
            raise OrchidDecisionError(
                "decision provider is closed",
                code="not_configured",
                provider=self._provider,
                retryable=False,
                host=self._host,
            )
        if not self._breaker.allow():
            raise OrchidDecisionError(
                "provider breaker is open",
                code="provider_unavailable",
                provider=self._provider,
                retryable=False,
                host=self._host,
            )

        loop = asyncio.get_running_loop()
        budget = self._timeout if timeout is None else max(0.1, float(timeout))
        deadline = loop.time() + budget
        attempts = self._max_retries + 1
        last_error: OrchidDecisionError | None = None

        try:
            await self._check_peer()
            for attempt in range(1, attempts + 1):
                if attempt > 1 and self._breaker.is_open:
                    raise last_error or self._breaker_error()

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise OrchidDecisionError(
                        "provider timed out",
                        code="timeout",
                        provider=self._provider,
                        retryable=True,
                        host=self._host,
                    )

                try:
                    async with asyncio.timeout(remaining):
                        status, headers, body = await self._send(path, payload)
                except TimeoutError:
                    last_error = OrchidDecisionError(
                        "provider timed out",
                        code="timeout",
                        provider=self._provider,
                        retryable=True,
                        host=self._host,
                    )
                    if attempt >= attempts:
                        self._breaker.record_failure()
                        raise last_error from None
                    if not await self._wait(attempt, None, deadline):
                        self._breaker.record_failure()
                        raise last_error from None
                    continue

                if status == 429 or 500 <= status < 600:
                    last_error = OrchidDecisionError(
                        f"provider returned HTTP {status}",
                        code="provider_unavailable",
                        provider=self._provider,
                        retryable=True,
                        status_code=status,
                        host=self._host,
                    )
                    if attempt >= attempts:
                        self._breaker.record_failure()
                        raise last_error
                    if not await self._wait(attempt, headers.get("retry-after"), deadline):
                        self._breaker.record_failure()
                        raise last_error
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
                    data = json.loads(body, parse_constant=_reject_json_constant)
                except (ValueError, RecursionError):
                    self._breaker.record_failure()
                    raise OrchidDecisionError(
                        "provider returned invalid JSON",
                        code="invalid_response",
                        provider=self._provider,
                        retryable=False,
                        host=self._host,
                    ) from None
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
        finally:
            self._breaker.abandon()

        assert last_error is not None  # pragma: no cover — the loop always returns or raises
        raise last_error  # pragma: no cover

    async def aclose(self) -> None:
        self._closed = True
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    # ── Internals ────────────────────────────────────────────

    def _breaker_error(self) -> OrchidDecisionError:
        return OrchidDecisionError(
            "provider breaker is open",
            code="provider_unavailable",
            provider=self._provider,
            retryable=False,
            host=self._host,
        )

    async def _check_peer(self) -> None:
        try:
            resolved = self._resolver(self._host)
            addresses = await resolved if inspect.isawaitable(resolved) else resolved
        except Exception:
            raise OrchidDecisionError(
                "provider host could not be resolved",
                code="provider_unavailable",
                provider=self._provider,
                retryable=True,
                host=self._host,
            ) from None
        if not addresses:
            raise OrchidDecisionError(
                "provider host resolved to no addresses",
                code="egress_denied",
                provider=self._provider,
                retryable=False,
                host=self._host,
            )
        for address in addresses:
            try:
                validate_resolved_address(address, label=f"provider {self._provider!r}")
            except ValueError as exc:
                raise OrchidDecisionError(
                    str(exc), code="egress_denied", provider=self._provider, retryable=False
                ) from None

    async def _wait(self, attempt: int, retry_after: str | None, deadline: float) -> bool:
        delay = self._backoff(attempt, retry_after)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= delay:
            return False
        await self._sleep(delay)
        return True

    def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise OrchidDecisionError(
                "decision provider is closed",
                code="not_configured",
                provider=self._provider,
                retryable=False,
                host=self._host,
            )
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
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _send(self, path: str, payload: dict[str, Any]) -> tuple[int, httpx.Headers, bytes]:
        client = self._get_client()
        try:
            async with client.stream("POST", self._url(path), json=payload, headers=self._headers()) as response:
                status = response.status_code
                if 300 <= status < 400:
                    raise OrchidDecisionError(
                        "provider attempted a redirect",
                        code="egress_denied",
                        provider=self._provider,
                        retryable=False,
                        status_code=status,
                        host=self._host,
                    )
                encoding = response.headers.get("content-encoding", "").strip().lower()
                if encoding and encoding != "identity":
                    raise OrchidDecisionError(
                        "provider returned a compressed response",
                        code="invalid_response",
                        provider=self._provider,
                        retryable=False,
                        status_code=status,
                        host=self._host,
                    )
                if status >= 400:
                    return status, response.headers, b""
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
                return status, response.headers, b"".join(chunks)
        except OrchidDecisionError:
            raise
        except httpx.TimeoutException:
            raise OrchidDecisionError(
                "provider timed out",
                code="timeout",
                provider=self._provider,
                retryable=True,
                host=self._host,
            ) from None
        except (httpx.DecodingError, httpx.TooManyRedirects):
            raise OrchidDecisionError(
                "provider returned an invalid response",
                code="invalid_response",
                provider=self._provider,
                retryable=False,
                host=self._host,
            ) from None
        except httpx.TransportError:
            raise OrchidDecisionError(
                "provider transport failed",
                code="provider_unavailable",
                provider=self._provider,
                retryable=True,
                host=self._host,
            ) from None
        except httpx.HTTPError:
            raise OrchidDecisionError(
                "provider request failed",
                code="provider_unavailable",
                provider=self._provider,
                retryable=False,
                host=self._host,
            ) from None
        except Exception:
            raise OrchidDecisionError(
                "provider request failed",
                code="provider_unavailable",
                provider=self._provider,
                retryable=False,
                host=self._host,
            ) from None

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            seconds = _parse_retry_after(retry_after)
            if seconds is not None:
                return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)
        base = min(0.25 * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
        return base * (0.75 + 0.25 * min(max(self._jitter(), 0.0), 1.0))


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _parse_retry_after(value: str) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if math.isfinite(seconds) else None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
