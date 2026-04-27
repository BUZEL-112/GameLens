"""
FastAPI Application Entry Point

Usage:
    cd game_recommender
    uvicorn recommendation_api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import pickle
import os
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI

from recommendation_api.core.config import settings
from recommendation_api.core.feature_store import FeatureStore
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


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global fs, retrieval_svc, reranking_svc, nearline

#     print("Starting up...")
#     fs = FeatureStore(
#         host=settings.redis_host,
#         port=settings.redis_port,
#         db=settings.redis_db,
#         password=settings.redis_password,
#     )
#     retrieval_svc = RetrievalService(settings.artifacts_path, fs)

#     artifacts = pickle.load(
#         open(f"{settings.artifacts_path}/artifacts.pkl", "rb")
#     )
#     reranking_svc = ReRankingService(
#         fs, artifacts, max_genres_per_response=settings.max_genres_per_response
#     )
#     nearline = NearlineUpdater(fs, retrieval_svc)
#     nearline.start()
#     print("API ready")

#     yield

#     print("Shutting down...")
#     nearline.stop()

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

    artifacts = pickle.load(
        open(f"{settings.artifacts_path}/artifacts.pkl", "rb")
    )

    # Auto-populate Redis if empty (first boot or after volume wipe)
    if fs.r.dbsize() == 0:
        print("Redis is empty — populating from artifacts...")
        _populate_redis(fs, artifacts)
        print("Redis population complete.")

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


def _populate_redis(fs: FeatureStore, artifacts: dict):
    import pandas as pd
    from training.utils.utils import load_config
    from training.pipeline import stage1_split_sides

    config = load_config("configs/config.yaml")
    data_path = os.environ.get(
        "DATA_PATH",
        config["feature_engineering"]["transformed_data_path"]
    )

    item_embeddings = np.load(f"{settings.artifacts_path}/item_embeddings.npy")

    with open(f"{settings.artifacts_path}/similarity_table.pkl", "rb") as f:
        similarity_table = pickle.load(f)

    df = pd.read_csv(data_path)
    items_df, interactions_df = stage1_split_sides(df)

    pop_totals = (
        interactions_df.groupby("item_name")["interaction"]
        .sum().sort_values(ascending=False)
    )
    popular_items = [
        {"item_name": name, "score": float(pop_totals[name]), "source": "popularity"}
        for name in pop_totals.index[:50]
    ]

    fs.populate_from_artifacts(
        artifacts=artifacts,
        item_embeddings=item_embeddings,
        all_user_vectors=artifacts["all_user_vectors"],
        similarity_table=similarity_table,
        items_df=items_df,
        interactions_df=interactions_df,
        popular_items=popular_items,
    )
app = FastAPI(
    title="Game Recommendation API",
    description="Two-Tower retrieval with Redis feature store",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommendations.router)
app.include_router(events.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=fs.get_model_version(),
        redis_keys=fs.r.dbsize(),
    )
