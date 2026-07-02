"""
Unit tests for recommendation_api.routers.recommendations

Strategy: build a minimal FastAPI app that mounts only the recommendations
router, then inject mock singletons into recommendation_api.main so the
router's `from recommendation_api.main import fs, retrieval_svc, reranking_svc`
calls resolve to controlled test doubles.

No Redis, no TensorFlow model, no FAISS index required.
Run with:
    pytest tests/api/test_recommendations_router.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recommendation_api.routers import recommendations


# ---------------------------------------------------------------------------
# Helpers to build a lightweight test app that bypasses APIKeyMiddleware
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    """
    Return a bare FastAPI instance mounting only the recommendations router.
    Middleware (APIKeyMiddleware) is intentionally omitted -- that logic is
    covered separately in test_security.py.
    """
    app = FastAPI()
    app.include_router(recommendations.router)
    return app


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

def _mock_fs(model_version: str = "v_test", popular: list | None = None) -> MagicMock:
    fs = MagicMock()
    fs.get_model_version.return_value = model_version
    fs.get_popular_items.return_value = popular or [
        {"item_name": "Counter-Strike", "score": 0.95},
        {"item_name": "Dota 2", "score": 0.90},
    ]
    return fs


def _mock_retrieval(
    candidates: list | None = None,
    similar: list | None = None,
) -> MagicMock:
    svc = MagicMock()
    svc.retrieve_candidates.return_value = candidates if candidates is not None else []
    svc.retrieve_similar_items.return_value = similar if similar is not None else []
    return svc


def _mock_reranking(reranked: list | None = None) -> MagicMock:
    svc = MagicMock()
    svc.rerank.return_value = reranked or [
        {"item_name": "Portal", "score": 0.88, "reason": None, "boosted": False},
    ]
    return svc


# ---------------------------------------------------------------------------
# Fixture: inject mocks before each test
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_with_mocks():
    """
    Yields a (TestClient, fs_mock, retrieval_mock, reranking_mock) tuple.
    Callers can reconfigure the mocks before issuing requests.
    """
    fs = _mock_fs()
    retrieval = _mock_retrieval()
    reranking = _mock_reranking()

    # The routers do `from recommendation_api.main import fs, retrieval_svc, ...`
    # at *call time* (deferred import inside the handler), so patching the
    # module-level names in recommendation_api.main is sufficient.
    with (
        patch("recommendation_api.main.fs", fs),
        patch("recommendation_api.main.retrieval_svc", retrieval),
        patch("recommendation_api.main.reranking_svc", reranking),
    ):
        app = _make_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, fs, retrieval, reranking


# ---------------------------------------------------------------------------
# Tests: popularity fallback (unknown user, no candidates)
# ---------------------------------------------------------------------------

class TestPopularityFallback:

    def test_returns_200_with_popular_items(self, client_with_mocks):
        client, fs, retrieval, _ = client_with_mocks
        # retrieval returns no candidates -> triggers popularity fallback
        retrieval.retrieve_candidates.return_value = []
        fs.get_popular_items.return_value = [
            {"item_name": "Counter-Strike", "score": 0.99},
            {"item_name": "Dota 2", "score": 0.90},
        ]

        resp = client.get("/v1/recommendations", params={"user_id": "unknown_user"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "popularity_fallback"
        assert len(body["recommendations"]) == 2
        assert body["recommendations"][0]["item_name"] == "Counter-Strike"
        assert body["recommendations"][0]["reason"] == "popular"

    def test_fallback_respects_count_parameter(self, client_with_mocks):
        client, fs, retrieval, _ = client_with_mocks
        retrieval.retrieve_candidates.return_value = []
        # Feature store returns 10 items; client requests only 3
        fs.get_popular_items.return_value = [
            {"item_name": f"Game {i}", "score": 1.0 - i * 0.05}
            for i in range(10)
        ]

        resp = client.get(
            "/v1/recommendations", params={"user_id": "ghost", "count": 3}
        )

        assert resp.status_code == 200
        # get_popular_items was called with top_k=3
        fs.get_popular_items.assert_called_once_with(top_k=3)

    def test_model_version_included_in_response(self, client_with_mocks):
        client, fs, retrieval, _ = client_with_mocks
        retrieval.retrieve_candidates.return_value = []
        fs.get_model_version.return_value = "v2.5"

        resp = client.get("/v1/recommendations", params={"user_id": "u1"})

        assert resp.json()["model_version"] == "v2.5"

    def test_latency_ms_is_non_negative(self, client_with_mocks):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_candidates.return_value = []

        resp = client.get("/v1/recommendations", params={"user_id": "u1"})

        assert resp.json()["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: personalized path (known user, candidates available)
# ---------------------------------------------------------------------------

class TestPersonalizedRecommendations:

    def test_returns_reranked_results(self, client_with_mocks):
        client, _, retrieval, reranking = client_with_mocks
        retrieval.retrieve_candidates.return_value = [
            {"item_name": "Portal", "score": 0.9, "source": "model"},
            {"item_name": "Half-Life", "score": 0.85, "source": "model"},
        ]
        reranking.rerank.return_value = [
            {"item_name": "Portal", "score": 0.9, "reason": None, "boosted": False},
        ]

        resp = client.get("/v1/recommendations", params={"user_id": "user_123"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "model"
        assert body["recommendations"][0]["item_name"] == "Portal"

    def test_rerank_receives_correct_user_id_and_count(self, client_with_mocks):
        client, _, retrieval, reranking = client_with_mocks
        retrieval.retrieve_candidates.return_value = [
            {"item_name": "Game A", "score": 0.8, "source": "model"},
        ]
        reranking.rerank.return_value = []

        client.get(
            "/v1/recommendations",
            params={"user_id": "alice", "count": 5, "context": "cart"},
        )

        reranking.rerank.assert_called_once_with(
            candidates=retrieval.retrieve_candidates.return_value,
            user_id="alice",
            top_k=5,
            context="cart",
        )

    def test_boosted_flag_propagated(self, client_with_mocks):
        client, _, retrieval, reranking = client_with_mocks
        retrieval.retrieve_candidates.return_value = [
            {"item_name": "Game B", "score": 0.7, "source": "model"},
        ]
        reranking.rerank.return_value = [
            {
                "item_name": "Game B",
                "score": 0.735,
                "reason": "cart_boost",
                "boosted": True,
            }
        ]

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "bob", "context": "cart"},
        )

        item = resp.json()["recommendations"][0]
        assert item["boosted"] is True
        assert item["reason"] == "cart_boost"


# ---------------------------------------------------------------------------
# Tests: item-to-item mode
# ---------------------------------------------------------------------------

class TestItemToItemMode:

    def test_returns_similar_items(self, client_with_mocks):
        client, fs, retrieval, _ = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "Dota 2", "score": 0.92},
            {"item_name": "Heroes of the Storm", "score": 0.87},
        ]
        fs.get_model_version.return_value = "v1.0"

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "item_name": "Counter-Strike"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "item_similarity"
        assert body["recommendations"][0]["item_name"] == "Dota 2"

    def test_reason_encodes_seed_item(self, client_with_mocks):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "Dota 2", "score": 0.92},
        ]

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "item_name": "Counter-Strike"},
        )

        reason = resp.json()["recommendations"][0]["reason"]
        assert reason == "similar_to_Counter-Strike"

    def test_reason_replaces_spaces_with_underscores(self, client_with_mocks):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "Portal 2", "score": 0.88},
        ]

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "item_name": "Half Life 2"},
        )

        reason = resp.json()["recommendations"][0]["reason"]
        assert " " not in reason
        assert "Half_Life_2" in reason

    def test_unknown_item_returns_404(self, client_with_mocks):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_similar_items.return_value = []

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "item_name": "NonExistentGame9999"},
        )

        assert resp.status_code == 404
        assert "NonExistentGame9999" in resp.json()["detail"]

    def test_retrieve_similar_called_with_correct_top_k(self, client_with_mocks):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "X", "score": 0.5},
        ]

        client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "item_name": "Portal", "count": 7},
        )

        retrieval.retrieve_similar_items.assert_called_once_with("Portal", top_k=7)


# ---------------------------------------------------------------------------
# Tests: query parameter validation
# ---------------------------------------------------------------------------

class TestQueryValidation:

    def test_count_below_minimum_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.get(
            "/v1/recommendations", params={"user_id": "u1", "count": 0}
        )
        assert resp.status_code == 422

    def test_count_above_maximum_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.get(
            "/v1/recommendations", params={"user_id": "u1", "count": 101}
        )
        assert resp.status_code == 422

    def test_invalid_context_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "context": "invalid_context"},
        )
        assert resp.status_code == 422

    def test_missing_user_id_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.get("/v1/recommendations")
        assert resp.status_code == 422

    @pytest.mark.parametrize("context", ["homepage", "product_page", "cart", "email"])
    def test_all_valid_contexts_accepted(self, client_with_mocks, context):
        client, _, retrieval, _ = client_with_mocks
        retrieval.retrieve_candidates.return_value = []

        resp = client.get(
            "/v1/recommendations",
            params={"user_id": "u1", "context": context},
        )
        assert resp.status_code == 200
