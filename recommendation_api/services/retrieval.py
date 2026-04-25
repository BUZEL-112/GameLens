"""
Retrieval Service

Hot path: user_id -> user embedding -> FAISS ANN search -> candidate list.

Three-tier embedding lookup:
    1. Cached embedding in Redis         (fastest -- zero model call)
    2. Raw features in Redis -> U-Tower  (medium -- one forward pass)
    3. Neither found                     (cold start -- return empty)
"""

from __future__ import annotations

import pickle
from typing import Optional

import faiss
import numpy as np
import tensorflow as tf

from recommendation_api.core.feature_store import FeatureStore


class RetrievalService:

    def __init__(self, artifacts_path: str, fs: FeatureStore):
        self.fs = fs

        art = pickle.load(open(f"{artifacts_path}/artifacts.pkl", "rb"))
        self.item_vocab = art["item_vocab"]
        self.idx_to_name = art["idx_to_name"]
        self.num_items = art["num_items"]
        self.USER_FEAT_DIM = art["USER_FEAT_DIM"]

        self.u_tower = tf.keras.models.load_model(f"{artifacts_path}/u_tower.keras")
        self.i_tower = tf.keras.models.load_model(f"{artifacts_path}/i_tower.keras")
        self.index = faiss.read_index(f"{artifacts_path}/item_index.faiss")
        print(f"RetrievalService ready -- {self.index.ntotal:,} items indexed")

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        # Fast path: cached embedding
        emb = self.fs.get_user_embedding(user_id)
        if emb is not None:
            return emb

        # Medium path: recompute from raw features and cache
        feats = self.fs.get_user_features(user_id)
        if feats is not None:
            emb = self.u_tower(
                feats.reshape(1, -1).astype(np.float32), training=False
            ).numpy()[0]
            self.fs.set_user_embedding(user_id, emb)
            return emb

        return None  # cold start

    def retrieve_candidates(
        self, user_id: str, n_candidates: int = 100
    ) -> list[dict]:
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return []
        query = np.ascontiguousarray(user_emb.reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(query, n_candidates)
        return [
            {
                "item_name": self.idx_to_name[int(i)],
                "score": float(s),
                "source": "model",
            }
            for i, s in zip(indices[0], scores[0])
            if int(i) in self.idx_to_name
        ]

    def retrieve_similar_items(
        self, item_name: str, top_k: int = 10
    ) -> list[dict]:
        # Try cache first
        cached = self.fs.get_similar_items(item_name)
        if cached:
            return cached[:top_k]

        if item_name not in self.item_vocab:
            return []

        item_emb = self.fs.get_item_embedding(item_name)
        if item_emb is None:
            return []

        query = np.ascontiguousarray(item_emb.reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(query, top_k + 1)
        results = [
            {"item_name": self.idx_to_name[int(i)], "score": float(s)}
            for i, s in zip(indices[0], scores[0])
            if int(i) in self.idx_to_name
            and self.idx_to_name[int(i)] != item_name
        ][:top_k]

        # Cache for next time
        self.fs.set_similar_items(item_name, results)
        return results
