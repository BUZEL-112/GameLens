"""
Weekly Retraining Pipeline

Re-runs the full training pipeline on the latest data, produces new
model artifacts, pushes embeddings and similarity tables to Redis,
and tags the new model version.

Usage:
    cd game_recommender
    python -m pipelines.retrain
    python -m pipelines.retrain --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

from training.utils.utils import load_config
from training.utils.logger import logger


def main(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    save_dir = config.get("serving", {}).get("artifacts_path", "model_artifacts")

    version_tag = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"=== Retraining Pipeline — {version_tag} ===")

    # ── Step 1: Run the training pipeline ─────────────────────────────────────
    from training.train import main as train_main

    logger.info("Running training pipeline...")
    train_main(config_path=config_path, skip_download=True)

    # ── Step 2: Load newly produced artifacts ─────────────────────────────────
    logger.info("Loading new artifacts...")
    with open(f"{save_dir}/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    item_embeddings = np.load(f"{save_dir}/item_embeddings.npy")

    with open(f"{save_dir}/similarity_table.pkl", "rb") as f:
        similarity_table = pickle.load(f)

    # ── Step 3: Push to Redis ─────────────────────────────────────────────────
    logger.info("Pushing artifacts to Redis...")
    try:
        from recommendation_api.core.config import settings
        from recommendation_api.core.feature_store import FeatureStore

        fs = FeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )

        # Load source data for populate call
        feat_cfg = config.get("feature_engineering", {})
        from training.pipeline import stage1_split_sides

        df = pd.read_csv(feat_cfg.get("transformed_data_path"))
        items_df, interactions_df = stage1_split_sides(df)

        # Compute popular items
        pop_totals = (
            interactions_df.groupby("item_name")["interaction"]
            .sum()
            .sort_values(ascending=False)
        )
        popular_items = [
            {"item_name": name, "score": float(pop_totals[name]),
             "source": "popularity"}
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
        fs.set_model_version(version_tag)
        fs.memory_report()

    except RuntimeError as e:
        logger.warning(f"Redis not available — skipping push: {e}")

    logger.info(f"Retraining pipeline complete — model version: {version_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly retraining job")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
