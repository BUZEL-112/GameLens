"""
Redis Initialization Script

Performs the initial population of the Redis feature store from training
artifacts. This is the first step after a fresh deployment or after wiping
the Redis volume. It must run to completion before starting the API container.

What it does:
1. Connects to Redis using the same settings as the API.
2. Loads all training artifacts from disk (embeddings, similarity table, etc.).
3. Loads the transformed data CSV and computes the popularity baseline.
4. Writes everything to Redis via populate_from_artifacts().
5. Sets the system:ready sentinel key, signaling that Redis is fully populated
   and the API is safe to start.

Usage:
    cd game_recommender
    python -m scripts.init_redis
    python -m scripts.init_redis --config configs/config.yaml

Prerequisites:
    - Training must have been run (model_artifacts/ populated)
    - Redis must be running (docker-compose up redis)
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from training.utils.utils import load_config
from training.utils.logger import logger


def main(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    save_dir = config.get("serving", {}).get("artifacts_path", "model_artifacts")

    logger.info("=== Redis Initialization ===")

    # ── Step 1: Connect to Redis ──────────────────────────────────────────────
    from recommendation_api.core.config import settings
    from recommendation_api.core.feature_store import FeatureStore

    fs = FeatureStore(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
    )

    # Guard: warn if Redis already has data
    existing_keys = fs.r.dbsize()
    if existing_keys > 0:
        logger.warning(
            f"Redis already contains {existing_keys:,} keys. "
            "This script will overwrite existing data."
        )

    # ── Step 2: Load training artifacts ───────────────────────────────────────
    logger.info(f"Loading artifacts from {save_dir}/")

    with open(f"{save_dir}/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    item_embeddings = np.load(f"{save_dir}/item_embeddings.npy")

    with open(f"{save_dir}/similarity_table.pkl", "rb") as f:
        similarity_table = pickle.load(f)

    # ── Step 3: Load transformed data and compute popularity ──────────────────
    feat_cfg = config.get("feature_engineering", {})
    data_path = os.environ.get(
        "DATA_PATH",
        feat_cfg.get("transformed_data_path", "data/processed/transformed_df.csv"),
    )
    logger.info(f"Loading transformed data from {data_path}")

    from training.pipeline import stage1_split_sides

    df = pd.read_csv(data_path)
    items_df, interactions_df = stage1_split_sides(df)

    pop_totals = (
        interactions_df.groupby("item_name")["interaction"]
        .sum()
        .sort_values(ascending=False)
    )
    popular_items = [
        {"item_name": name, "score": float(pop_totals[name]), "source": "popularity"}
        for name in pop_totals.index[:50]
    ]

    # ── Step 4: Populate Redis ────────────────────────────────────────────────
    logger.info("Writing all artifacts to Redis...")

    fs.populate_from_artifacts(
        artifacts=artifacts,
        item_embeddings=item_embeddings,
        all_user_vectors=artifacts["all_user_vectors"],
        similarity_table=similarity_table,
        items_df=items_df,
        interactions_df=interactions_df,
        popular_items=popular_items,
    )

    # ── Step 5: Set the readiness sentinel ────────────────────────────────────
    version_tag = "v1.0-init"
    fs.set_model_version(version_tag)
    fs.set_ready_sentinel(version_tag)
    fs.memory_report()

    logger.info(
        f"Redis initialization complete. "
        f"Sentinel key 'system:ready' set with version '{version_tag}'. "
        f"The API container is now safe to start."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize Redis feature store from training artifacts"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
