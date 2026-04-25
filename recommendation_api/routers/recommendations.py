"""
Recommendations router.

GET /v1/recommendations — personalized or item-to-item recommendations.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from recommendation_api.models.schemas import (
    ContextType,
    RecommendationResponse,
    RecommendedItem,
)

router = APIRouter(prefix="/v1", tags=["recommendations"])


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    count: int = Query(default=20, ge=1, le=100),
    context: ContextType = Query(default=ContextType.homepage),
    item_name: Optional[str] = Query(default=None),
):
    """
    Personalized recommendations for a user.

    - ``item_name`` present: item-to-item mode (no user profile needed)
    - Unknown user: popularity fallback
    - Known user: Two-Tower retrieval + reranking
    """
    from recommendation_api.main import fs, retrieval_svc, reranking_svc

    t0 = time.perf_counter()

    # ── Item-to-item mode ─────────────────────────────────────────────────────
    if item_name:
        similar = retrieval_svc.retrieve_similar_items(item_name, top_k=count)
        if not similar:
            raise HTTPException(
                status_code=404,
                detail=f"'{item_name}' not found or has no similar items",
            )
        return RecommendationResponse(
            recommendations=[
                RecommendedItem(
                    item_name=s["item_name"],
                    score=s["score"],
                    reason=f"similar_to_{item_name.replace(' ', '_')}",
                )
                for s in similar
            ],
            source="item_similarity",
            model_version=fs.get_model_version(),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Personalized mode ─────────────────────────────────────────────────────
    candidates = retrieval_svc.retrieve_candidates(user_id, n_candidates=count * 3)

    if not candidates:
        popular = fs.get_popular_items(top_k=count)
        return RecommendationResponse(
            recommendations=[
                RecommendedItem(
                    item_name=p["item_name"], score=p["score"], reason="popular"
                )
                for p in popular
            ],
            source="popularity_fallback",
            model_version=fs.get_model_version(),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    reranked = reranking_svc.rerank(
        candidates=candidates,
        user_id=user_id,
        top_k=count,
        context=context.value,
    )

    return RecommendationResponse(
        recommendations=[
            RecommendedItem(
                item_name=r["item_name"],
                score=r["score"],
                reason=r.get("reason"),
                boosted=r.get("boosted", False),
            )
            for r in reranked
        ],
        source="model",
        model_version=fs.get_model_version(),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
