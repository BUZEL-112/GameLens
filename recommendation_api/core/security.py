"""
Security layer for the recommendation API.

Provides two protections:
1. API Key Middleware — intercepts every request at the application level.
   Requests to any path under /v1/ must include a matching X-API-Key header.
   The /health endpoint is intentionally exempt (used by Docker's healthcheck
   which runs on the internal network without credentials).

2. Rate Limiter Dependency — a FastAPI dependency that can be attached to
   individual routes. Tracks request counts per caller (keyed by API key or
   client IP) within a sliding time window using an in-memory counter.
   Callers who exceed the limit receive a 429 Too Many Requests response.
   For a multi-instance deployment, swap the in-memory counter for a Redis
   INCR + EXPIRE approach using the existing FeatureStore Redis connection.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from recommendation_api.core.config import settings


# ── Layer 1: API Key Middleware ────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces X-API-Key authentication on all /v1/ routes.

    Exempt paths (no key required):
      - /health  — Docker internal healthcheck
      - /docs    — Swagger UI (disable in production if desired)
      - /openapi.json — schema endpoint used by Swagger
    """

    EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")

    async def dispatch(self, request: Request, call_next: Callable):
        # Allow exempt paths through without any key check
        for prefix in self.EXEMPT_PREFIXES:
            if request.url.path.startswith(prefix):
                return await call_next(request)

        # All /v1/ paths must provide a valid API key
        if request.url.path.startswith("/v1"):
            provided_key = request.headers.get("X-API-Key", "")
            if provided_key != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid X-API-Key header"},
                )

        return await call_next(request)


# ── Layer 2: Rate Limiter Dependency ──────────────────────────────────────────

class _InMemoryRateLimiter:
    """
    Sliding-window in-memory rate limiter.

    Tracks the timestamps of each request per caller identity. On each call,
    it prunes timestamps older than `window_seconds` and rejects the request
    if the remaining count is at or above `max_requests`.

    Thread-safe via a single reentrant lock — suitable for single-process
    deployments (single Uvicorn worker). For multi-worker deployments, replace
    with a Redis-backed counter.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # caller_id -> list of request timestamps
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, caller_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._windows[caller_id]
            # Slide the window: drop requests older than the cutoff
            self._windows[caller_id] = [t for t in timestamps if t > cutoff]
            if len(self._windows[caller_id]) >= self.max_requests:
                return False
            self._windows[caller_id].append(now)
            return True


# Module-level singleton — shared across all requests in this process
_events_limiter = _InMemoryRateLimiter(max_requests=100, window_seconds=60)


def events_rate_limit(request: Request) -> None:
    """
    FastAPI dependency that enforces per-caller rate limiting on the events endpoint.

    Caller identity is derived from the X-API-Key header (so all requests from
    the same integration count together), falling back to the client IP address
    for requests that somehow slip past the middleware.
    """
    caller_id = request.headers.get("X-API-Key") or request.client.host
    if not _events_limiter.is_allowed(caller_id):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {_events_limiter.max_requests} events "
                f"per {_events_limiter.window_seconds}s per caller"
            ),
        )
