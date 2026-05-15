"""
Data Validation Pipeline

Verifies that the data archive from data_retention.py was created successfully,
is readable via Pandas, and matches the expected row count.

Outputs a validation_manifest.json for the orchestrator.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from training.utils.utils import load_config
from training.utils.logger import logger


def main(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    manifest_dir = config.get("orchestration", {}).get("manifest_dir", "model_artifacts/manifests")
    
    retention_manifest_path = os.path.join(manifest_dir, "retention_manifest.json")
    
    logger.info("=== Data Validation Pipeline ===")
    
    if not os.path.exists(retention_manifest_path):
        raise FileNotFoundError(f"Missing retention manifest at {retention_manifest_path}")
        
    with open(retention_manifest_path, "r") as f:
        retention_data = json.load(f)
        
    archive_path = retention_data.get("archive_path")
    expected_rows = retention_data.get("events_archived", 0)
    
    is_valid = False
    error_msg = None
    
    if expected_rows > 0 and archive_path:
        logger.info(f"Validating archive: {archive_path}")
        if not os.path.exists(archive_path):
            error_msg = f"Archive file not found: {archive_path}"
        else:
            try:
                # Read just the metadata/first few rows to verify it's a valid Parquet
                # We don't need to load the whole file into memory, just checking readability
                df = pd.read_parquet(archive_path)
                actual_rows = len(df)
                if actual_rows != expected_rows:
                    error_msg = f"Row count mismatch! Expected {expected_rows}, got {actual_rows}"
                else:
                    logger.info("Parquet file is readable and row counts match.")
                    is_valid = True
            except Exception as e:
                error_msg = f"Failed to read Parquet file: {e}"
    else:
        logger.info("No events were archived in the previous run. Nothing to validate.")
        is_valid = True  # valid because there was nothing to do
        
    if not is_valid:
        logger.error(f"Validation FAILED: {error_msg}")
        raise RuntimeError(f"Data validation failed: {error_msg}")

    # Output validation manifest
    manifest = {
        "timestamp": time.time(),
        "is_valid": is_valid,
        "archive_path": archive_path,
        "expected_rows": expected_rows,
    }
    
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Data validation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validates the archived data")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
