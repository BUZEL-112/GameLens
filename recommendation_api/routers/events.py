"""
Events router.

POST /v1/events             — record a user interaction
GET  /v1/items/{name}/similar — item-to-item similarity lookup
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from recommendation_api.models.schemas import (
    EventResponse,
    SimilarItem,
    SimilarItemsResponse,
    UserEventRequest,
)

router = APIRouter(prefix="/v1", tags=["events"])


@router.post("/events", response_model=EventResponse, status_code=202)
async def record_event(event: UserEventRequest):
    """
    Record a user interaction. Returns 202 immediately -- processing is async.
    The NearlineUpdater refreshes the user's embedding within seconds.
    """
    from recommendation_api.main import fs

    # Synchronous: update the played set right away
    if event.event_type in ("purchase", "playtime", "click", "add_to_cart"):
        fs.add_played_item(str(event.user_id), event.item_name)

    # Async: push onto the event stream for embedding refresh
    fs.push_event(
        user_id=str(event.user_id),
        item_name=event.item_name,
        event_type=event.event_type.value,
        playtime=event.playtime,
        metadata=event.metadata,
    )

    return EventResponse(status="queued", timestamp=time.time())


@router.get("/items/{item_name}/similar", response_model=SimilarItemsResponse)
async def get_similar_items(item_name: str, count: int = 10):
    """Retrieve the top-N most similar games to the given game."""
    from recommendation_api.main import retrieval_svc

    similar = retrieval_svc.retrieve_similar_items(item_name, top_k=count)
    if not similar:
        raise HTTPException(status_code=404, detail=f"'{item_name}' not found")

    return SimilarItemsResponse(
        item_name=item_name,
        similar_items=[SimilarItem(**s) for s in similar],
    )
