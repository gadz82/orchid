"""Tests for the shared decision transport: caps, retries, breaker, egress."""

from __future__ import annotations

import json

import httpx
import pytest

from orchid_ai.core.decision import OrchidDecisionError
from orchid_ai.decision.http import DecisionBreaker, DecisionHttpClient

BASE_URL = "https://api.example.com"
HOSTS = ["api.example.com"]


def _http_client(handler, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client(handler, **kwargs) -> DecisionHttpClient:
    return DecisionHttpClient(
        base_url=BASE_URL,
        allowed_hosts=HOSTS,
        provider="fast",
        client=_http_client(handler),
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

    client = _client(handler, max_retries=1, sleep=fake_sleep)
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

    client = _client(handler, max_retries=1, sleep=fake_sleep)
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


async def test_connection_error_maps_to_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async def fake_sleep(seconds: float) -> None:
        return None

    client = _client(handler, max_retries=0, sleep=fake_sleep)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "provider_unavailable"


async def test_malformed_json_maps_to_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not-json")

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"


async def test_non_object_json_maps_to_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    client = _client(handler)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await client.request_json("/v1/decide", {})
    assert excinfo.value.code == "invalid_response"


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
        )
    assert excinfo.value.code == "egress_denied"
    assert calls["n"] == 0


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
    assert "breaker" in str(excinfo.value)
    assert calls["n"] == 2


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
    breaker.record_success()
    assert breaker.allow() is True


async def test_aclose_releases_the_client_reference() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _client(handler)
    await client.aclose()
    assert client._client is None
