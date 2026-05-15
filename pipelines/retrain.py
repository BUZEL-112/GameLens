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

    logger.info("=== Retraining Pipeline ===")

    # ── Step 1: Run the training pipeline ─────────────────────────────────────
    from training.train import main as train_main

    logger.info("Running training pipeline...")
    train_main(config_path=config_path, skip_download=True, frac_dat=False)

    # ── Step 2: Read Training Manifest ────────────────────────────────────────
    manifest_dir = config.get("orchestration", {}).get("manifest_dir", "model_artifacts/manifests")
    import json
    
    with open(os.path.join(manifest_dir, "training_manifest.json"), "r") as f:
        manifest = json.load(f)
        
    version_tag = manifest["version_tag"]
    logger.info(f"Retraining pipeline complete — model version: {version_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly retraining job")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
