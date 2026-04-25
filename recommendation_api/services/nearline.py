"""
Nearline Updater

Background daemon thread that consumes the Redis events:stream and
refreshes user embeddings after each new interaction.

The embedding is stale for at most UPDATE_INTERVAL seconds.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from recommendation_api.core.feature_store import FeatureStore


class NearlineUpdater(threading.Thread):

    UPDATE_INTERVAL = 5.0   # seconds between stream polls
    STREAM_KEY = "events:stream"
    BATCH_SIZE = 50         # events consumed per iteration

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
        entries = self.fs.r.xread(
            {self.STREAM_KEY: self._last_id},
            count=self.BATCH_SIZE,
            block=0,
        )
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
            feats = self.fs.get_user_features(uid)
            if feats is None:
                continue
            emb = self.retrieval_svc.u_tower(
                feats.reshape(1, -1).astype(np.float32), training=False
            ).numpy()[0]
            self.fs.set_user_embedding(uid, emb)

        if affected_users:
            print(
                f"[NearlineUpdater] refreshed {len(affected_users)} user embeddings"
            )
