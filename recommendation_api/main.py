"""
FastAPI Application Entry Point

Startup requires that Redis has already been populated via:
    python -m scripts.init_redis

If the system:ready sentinel key is missing from Redis, the application
will refuse to start with a clear error message.

Usage:
    cd game_recommender
    uvicorn recommendation_api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI

from recommendation_api.core.config import settings
from recommendation_api.core.feature_store import FeatureStore
from recommendation_api.core.security import APIKeyMiddleware
from recommendation_api.models.schemas import HealthResponse
from recommendation_api.routers import events, recommendations
from recommendation_api.services.nearline import NearlineUpdater
from recommendation_api.services.reranking import ReRankingService
from recommendation_api.services.retrieval import RetrievalService

# App-scope singletons -- imported by routers
fs: FeatureStore = None  # type: ignore[assignment]
retrieval_svc: RetrievalService = None  # type: ignore[assignment]
reranking_svc: ReRankingService = None  # type: ignore[assignment]
nearline: NearlineUpdater = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fs, retrieval_svc, reranking_svc, nearline

    print("Starting up...")
    fs = FeatureStore(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
    )

    # ── Sentinel check: refuse to start if Redis is not populated ─────────
    sentinel = fs.check_ready_sentinel()
    if sentinel is None:
        raise RuntimeError(
            "\n"
            "============================================================\n"
            "  Redis is not populated.\n"
            "  Run the init script before starting the API:\n"
            "\n"
            "      python -m scripts.init_redis\n"
            "\n"
            "  This populates Redis with training artifacts and sets the\n"
            "  system:ready sentinel key. The API will not start without it.\n"
            "============================================================"
        )
    print(
        f"Redis sentinel OK — model={sentinel['model_version']}, "
        f"populated_at={sentinel['populated_at']}"
    )

    artifacts_path = f"{settings.artifacts_path}/artifacts.pkl"
    import os
    if not os.path.exists(artifacts_path):
        raise RuntimeError(
            f"Artifacts missing at {artifacts_path}. "
            "Ensure the model_artifacts volume is mounted."
        )

    artifacts = pickle.load(open(artifacts_path, "rb"))

    retrieval_svc = RetrievalService(settings.artifacts_path, fs)
    reranking_svc = ReRankingService(
        fs, artifacts, max_genres_per_response=settings.max_genres_per_response
    )
    nearline = NearlineUpdater(fs, retrieval_svc)
    nearline.start()
    print("API ready")

    yield

    print("Shutting down...")
    nearline.stop()


app = FastAPI(
    title="Game Recommendation API",
    description="Two-Tower retrieval with Redis feature store",
    version="1.0.0",
    lifespan=lifespan,
)

# Layer 1: API key authentication on all /v1/ routes
app.add_middleware(APIKeyMiddleware)

app.include_router(recommendations.router)
app.include_router(events.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=fs.get_model_version(),
        redis_keys=fs.r.dbsize(),
    )
