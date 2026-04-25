"""
Pydantic schemas for API requests and responses.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ContextType(str, Enum):
    homepage = "homepage"
    product_page = "product_page"
    cart = "cart"
    email = "email"


class EventType(str, Enum):
    click = "click"
    purchase = "purchase"
    add_to_cart = "add_to_cart"
    impression = "impression"
    playtime = "playtime"


# ── Requests ──────────────────────────────────────────────────────────────────

class UserEventRequest(BaseModel):
    user_id: str
    item_name: str
    event_type: EventType
    playtime: float = 0.0
    metadata: dict = {}


# ── Response atoms ────────────────────────────────────────────────────────────

class RecommendedItem(BaseModel):
    item_name: str
    score: float
    reason: Optional[str] = None
    boosted: bool = False


class SimilarItem(BaseModel):
    item_name: str
    score: float


# ── Responses ─────────────────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedItem]
    source: str
    model_version: str
    latency_ms: float
    next_page_token: Optional[str] = None


class SimilarItemsResponse(BaseModel):
    item_name: str
    similar_items: list[SimilarItem]


class EventResponse(BaseModel):
    status: str
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    redis_keys: int
