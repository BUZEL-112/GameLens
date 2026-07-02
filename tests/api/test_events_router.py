"""
Unit tests for recommendation_api.routers.events

Tests cover:
- POST /v1/events  -- record a user interaction (202 accepted)
- GET  /v1/items/{item_name}/similar -- item similarity lookup
- GET  /v1/items/search -- item name search

Strategy: mount only the events router on a bare FastAPI app and patch the
recommendation_api.main singletons so no external services are needed.

Run with:
    pytest tests/api/test_events_router.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recommendation_api.routers import events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(events.router)
    return app


def _mock_fs() -> MagicMock:
    fs = MagicMock()
    fs.add_played_item.return_value = None
    fs.push_event.return_value = None
    return fs


def _mock_retrieval(vocab: list[str] | None = None) -> MagicMock:
    svc = MagicMock()
    svc.item_vocab = vocab or ["Counter-Strike", "Dota 2", "Portal"]
    svc.retrieve_similar_items.return_value = [
        {"item_name": "Dota 2", "score": 0.92},
        {"item_name": "Portal", "score": 0.87},
    ]
    return svc


@pytest.fixture()
def client_with_mocks():
    fs = _mock_fs()
    retrieval = _mock_retrieval()

    with (
        patch("recommendation_api.main.fs", fs),
        patch("recommendation_api.main.retrieval_svc", retrieval),
    ):
        # Override rate-limit dependency so it never blocks during tests
        from recommendation_api.core.security import events_rate_limit

        app = _make_app()
        app.dependency_overrides[events_rate_limit] = lambda: None

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, fs, retrieval


# ---------------------------------------------------------------------------
# POST /v1/events
# ---------------------------------------------------------------------------

class TestRecordEvent:

    def _valid_payload(self, event_type: str = "purchase") -> dict:
        return {
            "user_id": "user_abc",
            "item_name": "Counter-Strike",
            "event_type": event_type,
            "playtime": 120.0,
            "metadata": {},
        }

    def test_returns_202_on_valid_event(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.post("/v1/events", json=self._valid_payload("purchase"))
        assert resp.status_code == 202

    def test_response_has_queued_status(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.post("/v1/events", json=self._valid_payload())
        body = resp.json()
        assert body["status"] == "queued"
        assert "timestamp" in body

    def test_purchase_event_calls_add_played_item(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("purchase"))
        fs.add_played_item.assert_called_once_with("user_abc", "Counter-Strike")

    def test_playtime_event_calls_add_played_item(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("playtime"))
        fs.add_played_item.assert_called_once_with("user_abc", "Counter-Strike")

    def test_click_event_calls_add_played_item(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("click"))
        fs.add_played_item.assert_called_once_with("user_abc", "Counter-Strike")

    def test_add_to_cart_event_calls_add_played_item(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("add_to_cart"))
        fs.add_played_item.assert_called_once_with("user_abc", "Counter-Strike")

    def test_impression_event_does_not_call_add_played_item(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("impression"))
        fs.add_played_item.assert_not_called()

    def test_push_event_always_called(self, client_with_mocks):
        client, fs, *_ = client_with_mocks
        client.post("/v1/events", json=self._valid_payload("impression"))
        fs.push_event.assert_called_once_with(
            user_id="user_abc",
            item_name="Counter-Strike",
            event_type="impression",
            playtime=120.0,
            metadata={},
        )

    @pytest.mark.parametrize("bad_user_id", [
        "",                     # empty string
        "a" * 129,              # too long (> 128 chars)
        "user with spaces",     # spaces not allowed
        "user@example.com",     # @ not allowed
        "user#tag",             # # not allowed
    ])
    def test_invalid_user_id_returns_422(self, client_with_mocks, bad_user_id):
        client, *_ = client_with_mocks
        payload = self._valid_payload()
        payload["user_id"] = bad_user_id
        resp = client.post("/v1/events", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("good_user_id", [
        "user123",
        "guest_abc123",
        "76561198000000001",     # Steam numeric ID
        "a-b_c",                 # hyphens and underscores allowed
        "A",                     # single character
    ])
    def test_valid_user_id_formats_accepted(self, client_with_mocks, good_user_id):
        client, *_ = client_with_mocks
        payload = self._valid_payload()
        payload["user_id"] = good_user_id
        resp = client.post("/v1/events", json=payload)
        assert resp.status_code == 202

    def test_invalid_event_type_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        payload = self._valid_payload()
        payload["event_type"] = "not_a_real_event"
        resp = client.post("/v1/events", json=payload)
        assert resp.status_code == 422

    def test_missing_required_fields_returns_422(self, client_with_mocks):
        client, *_ = client_with_mocks
        resp = client.post("/v1/events", json={"user_id": "u1"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/items/{item_name}/similar
# ---------------------------------------------------------------------------

class TestGetSimilarItems:

    def test_returns_200_with_similar_items(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "Dota 2", "score": 0.92},
            {"item_name": "Portal", "score": 0.87},
        ]

        resp = client.get("/v1/items/Counter-Strike/similar")

        assert resp.status_code == 200
        body = resp.json()
        assert body["item_name"] == "Counter-Strike"
        assert len(body["similar_items"]) == 2
        assert body["similar_items"][0]["item_name"] == "Dota 2"

    def test_returns_404_when_no_similar_items(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.retrieve_similar_items.return_value = []

        resp = client.get("/v1/items/UnknownGame/similar")

        assert resp.status_code == 404
        assert "UnknownGame" in resp.json()["detail"]

    def test_count_parameter_forwarded_to_service(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "X", "score": 0.5},
        ]

        client.get("/v1/items/Portal/similar", params={"count": 5})

        retrieval.retrieve_similar_items.assert_called_once_with("Portal", top_k=5)

    def test_similar_items_contain_score(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.retrieve_similar_items.return_value = [
            {"item_name": "Dota 2", "score": 0.99},
        ]

        resp = client.get("/v1/items/Counter-Strike/similar")

        item = resp.json()["similar_items"][0]
        assert "score" in item
        assert item["score"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# GET /v1/items/search
# ---------------------------------------------------------------------------

class TestSearchItems:

    def test_returns_matches_for_known_prefix(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.item_vocab = ["Counter-Strike", "Counter-Strike: GO", "Dota 2"]

        resp = client.get("/v1/items/search", params={"q": "counter"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "counter"
        assert "Counter-Strike" in body["matches"]
        assert "Counter-Strike: GO" in body["matches"]
        assert "Dota 2" not in body["matches"]

    def test_search_is_case_insensitive(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.item_vocab = ["Portal", "Portal 2"]

        resp = client.get("/v1/items/search", params={"q": "PORTAL"})

        body = resp.json()
        assert "Portal" in body["matches"]
        assert "Portal 2" in body["matches"]

    def test_limit_caps_results(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.item_vocab = [f"Game {i}" for i in range(50)]

        resp = client.get("/v1/items/search", params={"q": "game", "limit": 5})

        assert len(resp.json()["matches"]) <= 5

    def test_no_matches_returns_empty_list(self, client_with_mocks):
        client, _, retrieval = client_with_mocks
        retrieval.item_vocab = ["Portal", "Dota 2"]

        resp = client.get("/v1/items/search", params={"q": "zzz_no_match"})

        assert resp.status_code == 200
        assert resp.json()["matches"] == []
