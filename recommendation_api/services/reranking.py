"""
Re-Ranking Service

Business rules layer applied on top of raw FAISS candidates.

Rules applied in order:
    1. Filter already-played items
    2. Enforce genre diversity (max N items per genre)
    3. Boost contextual items (e.g. cart context)
    4. Return top_k
"""

from __future__ import annotations

from recommendation_api.core.feature_store import FeatureStore


class ReRankingService:

    def __init__(
        self,
        fs: FeatureStore,
        artifacts: dict,
        max_genres_per_response: int = 3,
    ):
        self.fs = fs
        self.item_vocab = artifacts["item_vocab"]
        self.max_genres_per_resp = max_genres_per_response

    def rerank(
        self,
        candidates: list[dict],
        user_id: str,
        top_k: int = 20,
        context: str = "homepage",
    ) -> list[dict]:

        played = self.fs.get_played_items(user_id)
        genre_count: dict[str, int] = {}
        results = []

        for c in candidates:
            name = c["item_name"]

            # Rule 1: skip already played
            if name in played:
                continue

            # Rule 2: genre diversity cap
            meta = self.fs.get_item_meta(name)
            genres = meta.get("genres", []) if meta else []
            primary = genres[0] if genres else "Unknown"

            if genre_count.get(primary, 0) >= self.max_genres_per_resp:
                continue
            genre_count[primary] = genre_count.get(primary, 0) + 1

            item = {**c, "reason": None, "boosted": False}

            # Rule 3: contextual boost
            if context == "cart" and primary in ("Action", "Indie"):
                item["score"] *= 1.05
                item["boosted"] = True

            results.append(item)

            if len(results) == top_k:
                break

        return results
