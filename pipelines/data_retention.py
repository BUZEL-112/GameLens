"""
Data Retention Pipeline

Daily job that:
    1. Archives old events from Redis stream to Parquet files
    2. Snapshots user profile vectors for auditing / rollback
    3. Trims the Redis event stream to keep it bounded

Usage:
    cd game_recommender
    python -m pipelines.data_retention
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from training.utils.utils import load_config
from training.utils.logger import logger


def main(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    pipe_cfg = config.get("pipelines", {})
    archive_dir = pipe_cfg.get("archive_dir", "data/archive")
    snapshot_dir = pipe_cfg.get("snapshot_dir", "data/user_snapshots")
    retention_days = pipe_cfg.get("retention_days", 90)

    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    logger.info(f"=== Data Retention Pipeline — {today} ===")

    # ── Step 1: Archive events from Redis stream ──────────────────────────────
    try:
        from recommendation_api.core.config import settings
        from recommendation_api.core.feature_store import FeatureStore

        fs = FeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )

        stream_len = fs.r.xlen("events:stream")
        logger.info(f"Event stream length: {stream_len:,}")

        if stream_len > 0:
            # Read all events
            events = fs.r.xrange("events:stream", count=100_000)
            rows = []
            for msg_id, data in events:
                rows.append({
                    "msg_id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    "user_id": data.get(b"user_id", b"").decode(),
                    "item_name": data.get(b"item_name", b"").decode(),
                    "event_type": data.get(b"event_type", b"").decode(),
                    "playtime": float(data.get(b"playtime", b"0")),
                    "metadata": data.get(b"metadata", b"{}").decode(),
                })

            if rows:
                events_df = pd.DataFrame(rows)
                parquet_path = os.path.join(archive_dir, f"events_{today}.parquet")
                events_df.to_parquet(parquet_path, index=False)
                logger.info(
                    f"Archived {len(events_df):,} events to {parquet_path}"
                )

                # Trim stream: keep only last 10k events
                if stream_len > 10_000:
                    fs.r.xtrim("events:stream", maxlen=10_000, approximate=True)
                    logger.info(f"Trimmed stream to ~10,000 entries")

        # ── Step 2: Snapshot user feature vectors ─────────────────────────────
        logger.info("Snapshotting user feature vectors...")
        # Scan for user feature keys
        user_count = 0
        user_data = {}
        cursor = 0
        while True:
            cursor, keys = fs.r.scan(
                cursor=cursor, match="user:*:features", count=1000
            )
            for key in keys:
                uid = key.decode().split(":")[1]
                raw = fs.r.get(key)
                if raw:
                    vec = np.frombuffer(raw, dtype=np.float32).copy()
                    user_data[uid] = vec
                    user_count += 1
            if cursor == 0:
                break

        if user_data:
            snapshot_path = os.path.join(snapshot_dir, f"users_{today}.npz")
            np.savez_compressed(
                snapshot_path,
                user_ids=np.array(list(user_data.keys())),
                vectors=np.stack(list(user_data.values())),
            )
            logger.info(
                f"Snapshot saved: {user_count:,} users to {snapshot_path}"
            )

        # ── Step 3: Clean up old archives ─────────────────────────────────────
        logger.info(f"Cleaning archives older than {retention_days} days...")
        now = time.time()
        for dirname in [archive_dir, snapshot_dir]:
            for fname in os.listdir(dirname):
                fpath = os.path.join(dirname, fname)
                age_days = (now - os.path.getmtime(fpath)) / 86400
                if age_days > retention_days:
                    os.remove(fpath)
                    logger.info(f"  Removed {fpath} (age: {age_days:.0f} days)")

    except RuntimeError as e:
        logger.warning(f"Redis not available — skipping: {e}")

    logger.info("Data retention pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily data retention job")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
