"""
Unit tests for recommendation_api.core.security

Tests cover:
- APIKeyMiddleware: valid key, missing key, wrong key, exempt paths
- _InMemoryRateLimiter: allows traffic within limit, blocks on breach,
  respects sliding window, is thread-safe enough for concurrent callers

No Redis or model artifacts required.

Run with:
    pytest tests/api/test_security.py -v
"""

from __future__ import annotations

import time
from threading import Thread
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recommendation_api.core.security import (
    APIKeyMiddleware,
    _InMemoryRateLimiter,
    events_rate_limit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_KEY = "test-api-key-xyz"


def _make_secured_app(api_key: str = _TEST_KEY) -> FastAPI:
    """Minimal app with APIKeyMiddleware and a single /v1/ping route."""
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/v1/ping")
    def ping():
        return {"pong": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


@pytest.fixture()
def secured_client():
    with patch(
        "recommendation_api.core.security.settings"
    ) as mock_settings:
        mock_settings.api_key = _TEST_KEY
        app = _make_secured_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# APIKeyMiddleware
# ---------------------------------------------------------------------------

class TestAPIKeyMiddleware:

    def test_valid_key_allows_request(self, secured_client):
        resp = secured_client.get(
            "/v1/ping", headers={"X-API-Key": _TEST_KEY}
        )
        assert resp.status_code == 200

    def test_missing_key_returns_401(self, secured_client):
        resp = secured_client.get("/v1/ping")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, secured_client):
        resp = secured_client.get(
            "/v1/ping", headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 401

    def test_401_response_has_detail_field(self, secured_client):
        resp = secured_client.get("/v1/ping")
        assert "detail" in resp.json()

    def test_health_endpoint_exempt_from_key_check(self, secured_client):
        # /health must not require X-API-Key (Docker healthcheck)
        resp = secured_client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.parametrize("exempt_path", ["/docs", "/openapi.json", "/redoc"])
    def test_swagger_paths_exempt(self, secured_client, exempt_path):
        resp = secured_client.get(exempt_path)
        # May 200 or 404 depending on FastAPI config, but must NOT be 401
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# _InMemoryRateLimiter
# ---------------------------------------------------------------------------

class TestInMemoryRateLimiter:

    def test_allows_requests_within_limit(self):
        limiter = _InMemoryRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("caller_a") is True

    def test_blocks_request_after_limit_reached(self):
        limiter = _InMemoryRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("caller_b")
        assert limiter.is_allowed("caller_b") is False

    def test_different_callers_tracked_independently(self):
        limiter = _InMemoryRateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("alice")
        limiter.is_allowed("alice")
        # alice is blocked, bob should still be allowed
        assert limiter.is_allowed("alice") is False
        assert limiter.is_allowed("bob") is True

    def test_window_expiry_resets_count(self):
        limiter = _InMemoryRateLimiter(max_requests=2, window_seconds=1)
        limiter.is_allowed("caller_c")
        limiter.is_allowed("caller_c")
        assert limiter.is_allowed("caller_c") is False
        # Advance time past window
        with patch("recommendation_api.core.security.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 2
            # Rebuild a fresh limiter instance (simulating time advance)
        # Use a real sleep-free approach: inject a future timestamp via monkeypatching
        # Instead, verify via a new limiter with a tiny window + real sleep (pragmatic)
        limiter2 = _InMemoryRateLimiter(max_requests=1, window_seconds=1)
        limiter2.is_allowed("d")
        assert limiter2.is_allowed("d") is False
        time.sleep(1.1)
        assert limiter2.is_allowed("d") is True

    def test_concurrent_calls_stay_consistent(self):
        """Verify no race condition when many threads call is_allowed simultaneously."""
        limiter = _InMemoryRateLimiter(max_requests=50, window_seconds=60)
        results = []

        def worker():
            results.append(limiter.is_allowed("shared_caller"))

        threads = [Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed = sum(1 for r in results if r is True)
        blocked = sum(1 for r in results if r is False)
        assert allowed == 50
        assert blocked == 50
