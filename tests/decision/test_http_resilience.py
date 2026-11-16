"""Tests for the shared decision transport: caps, retries, breaker, egress."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from orchid_ai.core.decision import OrchidDecisionError
from orchid_ai.decision.http import DecisionBreaker, DecisionHttpClient

BASE_URL = "https://api.example.com"
HOSTS = ["api.example.com"]
PUBLIC_IP = "203.0.113.10"


def _resolver(host: str) -> list[str]:
    return [PUBLIC_IP]


def _http_client(handler, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


def _client(handler, *, resolver=_resolver, **kwargs) -> DecisionHttpClient:
    return DecisionHttpClient(
        base_url=BASE_URL,
        allowed_hosts=HOSTS,
        provider="fast",
        client=_http_client(handler),
        resolver=resolver,
        **kwargs,
    )


async def test_request_returns_decoded_object() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {}})

    client = _client(handler)
    result = await client.request_json("/v1/decide", {"state": "hi"})
    assert result == {"answers": {}}
    assert seen == [{"state": "hi"}]
    assert client.host == "api.example.com"


async def test_authorization_header_is_sent_when_key_present() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={})

    client = _client(handler, api_key="secret-value")
    await client.request_json("/v1/decide", {})
    assert seen == ["Bearer secret-value"]


async def test_identity_encoding_is_requested() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("accept-encoding", ""))
        return httpx.Response(200, json={})

    client = _client(handler)
    await client.request_json("/v1/decide", {})
    assert seen == ["identity"]


async def test_429_retries_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0.5"})
        return httpx.Response(200, json={"ok": True})

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = _client(handler, max_retries=1, sleep=fake_sleep, timeout=10.0)
    assert await client.request_json("/v1/decide", {}) == {"ok": True}
    assert calls["n"] == 2
    assert sleeps == [0.5]


async def test_5xx_exhaustion_raises_retryable_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(handler, max_retries=1, sleep=fake_sleep, timeout=10.0)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "provider_unavailable"
    assert excinfo.value.retryable is True
    assert excinfo.value.status_code == 503
    assert calls["n"] == 2


async def test_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad"})

    client = _client(handler, max_retries=3)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert calls["n"] == 1
    assert excinfo.value.retryable is False


async def test_timeout_maps_to_timeout_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(handler, max_retries=0, sleep=fake_sleep)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "timeout"
    assert excinfo.value.retryable is True


async def test_slow_provider_is_bounded_by_the_whole_call_deadline() -> None:
    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.5)
            return httpx.Response(200, json={})

    client = DecisionHttpClient(
        base_url=BASE_URL,
        allowed_hosts=HOSTS,
        provider="fast",
        client=httpx.AsyncClient(transport=SlowTransport()),
        resolver=_resolver,
        timeout=0.1,
    )
    started = time.monotonic()
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "timeout"
    assert time.monotonic() - started < 1.0


async def test_connection_error_maps_to_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(handler, max_retries=0, sleep=fake_sleep)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "provider_unavailable"


@pytest.mark.parametrize(
    "content",
    [
        b"{not-json",
        b'{"a": ' + b"9" * 5000 + b"}",
        b'{"a": ' + b'{"b": ' * 20000 + b"1" + b"}" * 20000 + b"}",
        b'{"a": NaN}',
        b'{"a": Infinity}',
    ],
)
async def test_invalid_json_maps_to_invalid_response(content: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"
    assert excinfo.value.__cause__ is None


async def test_non_object_json_maps_to_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"


async def test_compressed_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x1f\x8b", headers={"content-encoding": "gzip"})

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"


async def test_redirect_is_treated_as_egress_denied() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(307, headers={"location": "https://evil.example.com/steal"})

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "egress_denied"
    assert calls["n"] == 1


async def test_oversized_response_is_rejected_without_leaking_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"SECRET" * 1024)

    client = _client(handler, max_response_bytes=1024)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"
    assert "SECRET" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("base_url", "hosts"),
    [
        ("http://api.example.com", HOSTS),
        ("https://other.example.com", HOSTS),
        ("https://localhost", ["localhost"]),
        ("https://127.0.0.1", ["127.0.0.1"]),
        ("https://user:pass@api.example.com", HOSTS),
        ("https://api.example.com", []),
    ],
)
def test_egress_denial_happens_before_any_request(base_url: str, hosts: list[str]) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    with pytest.raises(OrchidDecisionError) as excinfo:
        DecisionHttpClient(
            base_url=base_url,
            allowed_hosts=hosts,
            provider="fast",
            client=_http_client(handler),
            resolver=_resolver,
        )
    assert excinfo.value.code == "egress_denied"
    assert calls["n"] == 0


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "::1", "fd00:ec2::254", "64:ff9b::7f00:1"])
async def test_resolved_metadata_or_loopback_address_is_denied(address: str) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    client = _client(handler, resolver=lambda host: [address])
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "egress_denied"
    assert calls["n"] == 0


async def test_injected_client_with_redirects_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with pytest.raises(OrchidDecisionError) as excinfo:
        DecisionHttpClient(
            base_url=BASE_URL,
            allowed_hosts=HOSTS,
            provider="fast",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True),
            resolver=_resolver,
        )
    assert excinfo.value.code == "egress_denied"


async def test_breaker_opens_after_threshold_and_fails_fast() -> None:
    calls = {"n": 0}
    now = {"t": 100.0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(
        handler,
        max_retries=0,
        sleep=fake_sleep,
        failure_threshold=2,
        cooldown_s=10.0,
        clock=lambda: now["t"],
    )
    for _ in range(2):
        with pytest.raises(OrchidDecisionError):
            await client.request_json("/v1/decide", {})
    assert calls["n"] == 2

    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "provider_unavailable"
    assert calls["n"] == 2


async def test_breaker_counts_one_failure_per_call_not_per_attempt() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(handler, max_retries=3, sleep=fake_sleep, timeout=10.0, failure_threshold=3)
    with pytest.raises(OrchidDecisionError):
        await client.request_json("/v1/decide", {})
    assert calls["n"] == 4
    assert client._breaker.failures == 1


async def test_breaker_half_open_after_cooldown_and_closes_on_success() -> None:
    calls = {"n": 0}
    now = {"t": 100.0}
    failing = {"on": True}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if failing["on"]:
            return httpx.Response(500)
        return httpx.Response(200, json={})

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(
        handler,
        max_retries=0,
        sleep=fake_sleep,
        failure_threshold=1,
        cooldown_s=10.0,
        clock=lambda: now["t"],
    )
    with pytest.raises(OrchidDecisionError):
        await client.request_json("/v1/decide", {})
    assert calls["n"] == 1

    now["t"] = 111.0
    failing["on"] = False
    assert await client.request_json("/v1/decide", {}) == {}
    assert calls["n"] == 2

    failing["on"] = True
    now["t"] = 200.0
    with pytest.raises(OrchidDecisionError):
        await client.request_json("/v1/decide", {})
    assert calls["n"] == 3


async def test_cancelled_half_open_probe_releases_the_breaker() -> None:
    now = {"t": 100.0}
    calls = {"n": 0}
    slow = {"on": False}

    def failing_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    def ok_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    async def resolver(host: str) -> list[str]:
        if slow["on"]:
            await asyncio.sleep(10)
        return [PUBLIC_IP]

    client = _client(
        failing_handler,
        resolver=resolver,
        max_retries=0,
        failure_threshold=1,
        cooldown_s=1.0,
        clock=lambda: now["t"],
    )
    with pytest.raises(OrchidDecisionError):
        await client.request_json("/v1/decide", {})

    now["t"] = 102.0
    slow["on"] = True
    task = asyncio.create_task(client.request_json("/v1/decide", {}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    slow["on"] = False
    client._client = _http_client(ok_handler)
    assert await client.request_json("/v1/decide", {}) == {}
    assert calls["n"] >= 2


def test_breaker_state_machine_without_io() -> None:
    now = {"t": 0.0}
    breaker = DecisionBreaker(failure_threshold=2, cooldown_s=5.0, clock=lambda: now["t"])
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is False

    now["t"] = 6.0
    assert breaker.allow() is True
    assert breaker.allow() is False  # one half-open probe at a time
    breaker.abandon()
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.allow() is True


async def test_aclose_marks_the_client_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _client(handler)
    await client.aclose()
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "not_configured"


async def test_retry_after_nan_falls_back_to_backoff() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "nan"})
        return httpx.Response(200, json={})

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = _client(handler, max_retries=1, sleep=fake_sleep, timeout=10.0, jitter=lambda: 0.0)
    assert await client.request_json("/v1/decide", {}) == {}
    assert len(sleeps) == 1
    assert 0.1 <= sleeps[0] <= 2.0
