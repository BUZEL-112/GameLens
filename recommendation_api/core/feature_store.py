"""
Redis Feature Store

Thin wrapper around Redis providing typed access to all feature data
required by the recommendation API.

Key schema:
    user:{uid}:embedding     bytes (float32)   7 days    128-dim user emb
    user:{uid}:features      bytes (float32)   7 days    322-dim raw vec
    user:{uid}:played        JSON string       30 days   set of item names
    item:{name}:embedding    bytes (float32)   permanent 128-dim item emb
    item:{name}:similar      JSON string       permanent top-10 neighbors
    item:{name}:meta         JSON string       permanent genres, tags
    global:popular           JSON string       1 hour    top-50 popular
    model:version            string            permanent current tag
    events:stream            Redis Stream      permanent incoming events
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import redis


class FeatureStore:

    TTL_USER_EMBEDDING = 60 * 60 * 24 * 7   # 7 days
    TTL_USER_PLAYED = 60 * 60 * 24 * 30     # 30 days
    TTL_POPULAR = 60 * 60                    # 1 hour

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = "",
    ):
        kwargs = dict(
            host=host,
            port=port,
            db=db,
            decode_responses=False,
            socket_timeout=2.0,
            socket_connect_timeout=5.0,
        )
        if password:
            kwargs["password"] = password

        self.r = redis.Redis(**kwargs)
        self._ping()

    def _ping(self):
        try:
            self.r.ping()
            info = self.r.info("server")
            print(
                f"Redis connected  (v{info['redis_version']}  "
                f"port={info['tcp_port']}  "
                f"used_memory={info.get('used_memory_human', 'Unknown')})"
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise RuntimeError(
                f"Cannot connect to Redis: {e}\n"
                "Start Redis first:\n"
                "  docker run -d --name redis-rec -p 6379:6379 redis:7"
            )

    # ── Embedding operations ──────────────────────────────────────────────────

    def set_user_embedding(self, user_id: str, embedding: np.ndarray):
        key = f"user:{user_id}:embedding"
        self.r.set(key, embedding.astype(np.float32).tobytes())
        self.r.expire(key, self.TTL_USER_EMBEDDING)

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        raw = self.r.get(f"user:{user_id}:embedding")
        return np.frombuffer(raw, dtype=np.float32).copy() if raw else None

    def set_user_features(self, user_id: str, features: np.ndarray):
        key = f"user:{user_id}:features"
        self.r.set(key, features.astype(np.float32).tobytes())
        self.r.expire(key, self.TTL_USER_EMBEDDING)

    def get_user_features(self, user_id: str) -> Optional[np.ndarray]:
        raw = self.r.get(f"user:{user_id}:features")
        return np.frombuffer(raw, dtype=np.float32).copy() if raw else None

    def set_item_embedding(self, item_name: str, embedding: np.ndarray):
        self.r.set(
            f"item:{item_name}:embedding",
            embedding.astype(np.float32).tobytes(),
        )

    def get_item_embedding(self, item_name: str) -> Optional[np.ndarray]:
        raw = self.r.get(f"item:{item_name}:embedding")
        return np.frombuffer(raw, dtype=np.float32).copy() if raw else None

    # ── Structured data ───────────────────────────────────────────────────────

    def set_played_items(self, user_id: str, played: set):
        key = f"user:{user_id}:played"
        self.r.set(key, json.dumps(list(played)))
        self.r.expire(key, self.TTL_USER_PLAYED)

    def get_played_items(self, user_id: str) -> set:
        raw = self.r.get(f"user:{user_id}:played")
        return set(json.loads(raw)) if raw else set()

    def add_played_item(self, user_id: str, item_name: str):
        current = self.get_played_items(user_id)
        current.add(item_name)
        self.set_played_items(user_id, current)

    def set_similar_items(self, item_name: str, similar: list):
        self.r.set(f"item:{item_name}:similar", json.dumps(similar))

    def get_similar_items(self, item_name: str) -> Optional[list]:
        raw = self.r.get(f"item:{item_name}:similar")
        return json.loads(raw) if raw else None

    def set_item_meta(self, item_name: str, meta: dict):
        self.r.set(f"item:{item_name}:meta", json.dumps(meta))

    def get_item_meta(self, item_name: str) -> Optional[dict]:
        raw = self.r.get(f"item:{item_name}:meta")
        return json.loads(raw) if raw else None

    def set_popular_items(self, popular: list):
        self.r.set("global:popular", json.dumps(popular))
        self.r.expire("global:popular", self.TTL_POPULAR)

    def get_popular_items(self, top_k: int = 20) -> list:
        raw = self.r.get("global:popular")
        return json.loads(raw)[:top_k] if raw else []

    def set_model_version(self, version: str):
        self.r.set("model:version", version.encode())

    def get_model_version(self) -> str:
        raw = self.r.get("model:version")
        return raw.decode() if raw else "unknown"

    # ── Readiness sentinel ────────────────────────────────────────────────────

    SENTINEL_KEY = "system:ready"

    def set_ready_sentinel(self, model_version: str):
        """
        Write the readiness sentinel after a successful Redis population.
        Stores the model version and a UTC timestamp so operators can audit
        when Redis was last fully populated and from which training run.
        """
        import json
        from datetime import datetime, timezone

        payload = json.dumps({
            "model_version": model_version,
            "populated_at": datetime.now(timezone.utc).isoformat(),
        })
        self.r.set(self.SENTINEL_KEY, payload.encode())

    def check_ready_sentinel(self) -> dict | None:
        """
        Returns the sentinel payload if Redis has been populated, else None.
        """
        import json

        raw = self.r.get(self.SENTINEL_KEY)
        if raw is None:
            return None
        return json.loads(raw)

    # ── Event stream ──────────────────────────────────────────────────────────

    def push_event(
        self,
        user_id: str,
        item_name: str,
        event_type: str,
        playtime: float = 0.0,
        metadata: dict | None = None,
    ):
        self.r.xadd(
            "events:stream",
            {
                "user_id": user_id,
                "item_name": item_name,
                "event_type": event_type,
                "playtime": str(playtime),
                "metadata": json.dumps(metadata or {}),
            },
        )

    # ── Bulk population ───────────────────────────────────────────────────────

    def populate_from_artifacts(
        self,
        artifacts: dict,
        item_embeddings: np.ndarray,
        all_user_vectors: dict,
        similarity_table: dict,
        items_df,
        interactions_df,
        popular_items: list,
        user_batch_size: int = 10_000,
    ):
        """One-time bulk load of all training artifacts into Redis."""
        pipe = self.r.pipeline(transaction=False)
        t0 = time.perf_counter()

        # Item embeddings
        print("  Writing item embeddings...")
        for name, idx in artifacts["item_vocab"].items():
            pipe.set(
                f"item:{name}:embedding",
                item_embeddings[idx].astype(np.float32).tobytes(),
            )
        pipe.execute()

        # Item metadata
        print("  Writing item metadata...")
        for _, row in items_df.iterrows():
            meta = {
                "genres": row["genres"],
                "tags": row["tags"][:20] if isinstance(row["tags"], list) else [],
                "in_stock": True,
            }
            pipe.set(f"item:{row['item_name']}:meta", json.dumps(meta))
        pipe.execute()

        # Similarity table
        print("  Writing similarity table...")
        for name, similars in similarity_table.items():
            pipe.set(f"item:{name}:similar", json.dumps(similars))
        pipe.execute()

        # User feature vectors
        print("  Writing user feature vectors...")
        uids = list(all_user_vectors.keys())
        for start in range(0, len(uids), user_batch_size):
            for uid in uids[start : start + user_batch_size]:
                vec = all_user_vectors[uid].astype(np.float32).tobytes()
                key = f"user:{uid}:features"
                pipe.set(key, vec)
                pipe.expire(key, self.TTL_USER_EMBEDDING)
            pipe.execute()

        # Played item sets
        print("  Writing played item sets...")
        for uid, group in interactions_df.groupby("user_id", sort=False):
            played = list(set(group["item_name"].tolist()))
            key = f"user:{uid}:played"
            pipe.set(key, json.dumps(played))
            pipe.expire(key, self.TTL_USER_PLAYED)
        pipe.execute()

        # Popular items + model version
        self.set_popular_items(popular_items)
        self.set_model_version("v1.0")

        elapsed = time.perf_counter() - t0
        n_keys = self.r.dbsize()
        print(
            f"\nRedis populated: {n_keys:,} total keys  "
            f"in {elapsed:.1f}s  ({n_keys / elapsed:.0f} keys/sec)"
        )

    # ── Memory report ─────────────────────────────────────────────────────────

    def memory_report(self):
        info = self.r.info("memory")
        print(f"\nRedis memory report:")
        print(f"  Used memory      : {info['used_memory_human']}")
        print(f"  Peak memory      : {info['used_memory_peak_human']}")
        print(f"  Total keys       : {self.r.dbsize():,}")
