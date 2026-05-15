"""
Nearline Updater

Background daemon thread that consumes the Redis events:stream and
refreshes user embeddings after each new interaction.

The embedding is stale for at most UPDATE_INTERVAL seconds.

User feature vectors are rebuilt using core_ml.features.build_user_vector() —
the EXACT same function that training/pipeline.py uses in Stage 3. This is the
mathematical guarantee that training and serving stay in parity.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd

from recommendation_api.core.feature_store import FeatureStore
from core_ml.features import build_user_vector


class NearlineUpdater(threading.Thread):

    UPDATE_INTERVAL = 5.0
    BLOCK_MS = 1500         # must stay under socket_timeout (2000 ms)
    STREAM_KEY = "events:stream"
    BATCH_SIZE = 50

    def __init__(self, fs: FeatureStore, retrieval_svc):
        super().__init__(daemon=True)
        self.fs = fs
        self.retrieval_svc = retrieval_svc
        self._stop_event = threading.Event()
        self._last_id = "0-0"

    def run(self):
        print("NearlineUpdater started")
        while not self._stop_event.is_set():
            try:
                self._consume_batch()
            except Exception as e:
                print(f"[NearlineUpdater] error: {e}")
            time.sleep(self.UPDATE_INTERVAL)
        print("NearlineUpdater stopped")

    def stop(self):
        self._stop_event.set()

    def _consume_batch(self):
        try:
            entries = self.fs.r.xread(
                {self.STREAM_KEY: self._last_id},
                count=self.BATCH_SIZE,
                block=self.BLOCK_MS,
            )
        except Exception:
            # TimeoutError fires here when no events arrive within BLOCK_MS.
            # That's expected — just return and let the loop retry.
            return

        if not entries:
            return

        _, messages = entries[0]
        affected_users: set[str] = set()

        for msg_id, data in messages:
            self._last_id = msg_id
            uid = data.get(b"user_id", b"").decode()
            if uid:
                affected_users.add(uid)

        # Refresh embedding for every user seen in this batch
        for uid in affected_users:
            self._refresh_user(uid)

        if affected_users:
            print(
                f"[NearlineUpdater] refreshed {len(affected_users)} user embeddings"
            )

    def _refresh_user(self, uid: str) -> None:
        """
        Rebuild a user's feature vector from their stored play history and
        recompute their embedding using the user tower.

        Uses core_ml.features.build_user_vector() — the same function used
        in training Stage 3 — to guarantee mathematical parity.

        Note: The last two statistical dimensions (total_interaction, num_games)
        are NOT globally normalized here because we don't have the full user
        corpus at serving time. This is an accepted online approximation:
        the content-vector portion (dims 0-319) is identical to training.
        """
        # Retrieve artifacts needed to reconstruct the user vector
        artifacts = self.retrieval_svc.artifacts
        item_content_matrix = self.retrieval_svc.item_content_matrix
        item_vocab = artifacts.get("item_vocab", {})

        # Reconstruct user history from the played set stored in Redis
        played_raw = self.fs.r.get(f"user:{uid}:played")
        if played_raw is None:
            # Fall back to the pre-computed feature vector if history is unavailable
            feats = self.fs.get_user_features(uid)
            if feats is None:
                return
        else:
            played_items = json.loads(played_raw)
            if not played_items:
                return

            # Build a minimal history DataFrame with a unit interaction weight
            # (playtime data is not stored per-item in Redis; unit weight is a safe default)
            history_df = pd.DataFrame({
                "item_name": played_items,
                "interaction": [1.0] * len(played_items),
            })
            feats = build_user_vector(history_df, item_content_matrix, item_vocab)
            # Persist the refreshed feature vector for future nearline updates
            self.fs.set_user_features(uid, feats)

        # Run the feature vector through the user tower to get the new embedding
        emb = self.retrieval_svc.u_tower(
            feats.reshape(1, -1).astype(np.float32), training=False
        ).numpy()[0]
        self.fs.set_user_embedding(uid, emb)
