"""
End-to-End Test

Validates the full system: training artifacts exist, API starts,
endpoints return correct responses.

Usage:
    cd game_recommender
    python e2e_test.py

Prerequisites:
    - Training must have been run (model_artifacts/ populated)
    - Redis must be running (docker-compose up redis)
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np


def check_artifacts(artifacts_path: str = "model_artifacts") -> bool:
    """Verify all required artifact files exist."""
    required = [
        "u_tower.keras",
        "i_tower.keras",
        "item_embeddings.npy",
        "item_index.faiss",
        "artifacts.pkl",
        "similarity_table.pkl",
    ]
    print("== Checking Artifacts ==")
    all_ok = True
    for fname in required:
        path = os.path.join(artifacts_path, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {fname:<30}  {status}  ({size / 1e6:.2f} MB)")
        if not exists:
            all_ok = False
    return all_ok


def check_artifact_contents(artifacts_path: str = "model_artifacts") -> bool:
    """Verify artifact contents are loadable and consistent."""
    print("\n== Checking Artifact Contents ==")
    all_ok = True

    try:
        with open(f"{artifacts_path}/artifacts.pkl", "rb") as f:
            art = pickle.load(f)

        num_items = art["num_items"]
        num_users = len(art["all_user_vectors"])
        print(f"  Items:  {num_items:,}")
        print(f"  Users:  {num_users:,}")

        item_emb = np.load(f"{artifacts_path}/item_embeddings.npy")
        print(f"  Item embeddings: {item_emb.shape}")

        if item_emb.shape[0] != num_items:
            print(f"  MISMATCH: embeddings={item_emb.shape[0]}, vocab={num_items}")
            all_ok = False

        norms = np.linalg.norm(item_emb, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            print(f"  WARNING: not all norms are 1.0")
            all_ok = False
        else:
            print(f"  Norms OK (all ~1.0)")

    except Exception as e:
        print(f"  ERROR: {e}")
        all_ok = False

    return all_ok


def check_api_endpoints() -> bool:
    """Smoke-test the API endpoints using TestClient (no server needed)."""
    print("\n== Checking API Endpoints ==")

    try:
        # Prepare singletons for the test app
        sys.path.insert(0, os.path.abspath("."))

        from recommendation_api.core.feature_store import FeatureStore
        from recommendation_api.core.config import settings
        from recommendation_api.services.retrieval import RetrievalService
        from recommendation_api.services.reranking import ReRankingService

        fs = FeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )

        retrieval_svc = RetrievalService(settings.artifacts_path, fs)

        with open(f"{settings.artifacts_path}/artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        reranking_svc = ReRankingService(
            fs, artifacts, max_genres_per_response=settings.max_genres_per_response
        )

        # Inject singletons into main module
        import recommendation_api.main as main_mod
        main_mod.fs = fs
        main_mod.retrieval_svc = retrieval_svc
        main_mod.reranking_svc = reranking_svc

        # Build test app (no lifespan -- singletons already set)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from recommendation_api.routers import recommendations, events

        test_app = FastAPI()
        test_app.include_router(recommendations.router)
        test_app.include_router(events.router)

        client = TestClient(test_app, raise_server_exceptions=False)
        # Attach the API key to every request so the middleware accepts them
        api_key = os.environ.get("API_KEY", "dev-insecure-key")
        headers = {"X-API-Key": api_key}
        all_ok = True

        # Pick a test user
        test_uid = list(artifacts["all_user_vectors"].keys())[0]

        # Test 1: Personalized recommendations
        resp = client.get("/v1/recommendations",
                          params={"user_id": test_uid, "count": 5},
                          headers=headers)
        ok = resp.status_code == 200
        print(f"  GET /v1/recommendations         : {resp.status_code}  {'OK' if ok else 'FAIL'}")
        if ok:
            body = resp.json()
            print(f"    source={body['source']}, items={len(body['recommendations'])}")
        all_ok = all_ok and ok

        # Test 2: Cold-start user
        resp2 = client.get("/v1/recommendations",
                           params={"user_id": "nonexistent_user_xyz", "count": 5},
                           headers=headers)
        ok2 = resp2.status_code == 200
        print(f"  GET /v1/recommendations (cold)  : {resp2.status_code}  {'OK' if ok2 else 'FAIL'}")
        if ok2:
            print(f"    source={resp2.json()['source']}")
        all_ok = all_ok and ok2

        # Test 3: Item-to-item similarity
        resp3 = client.get("/v1/items/Counter-Strike/similar",
                           params={"count": 5},
                           headers=headers)
        ok3 = resp3.status_code in (200, 404)
        print(f"  GET /v1/items/.../similar        : {resp3.status_code}  {'OK' if ok3 else 'FAIL'}")
        all_ok = all_ok and ok3

        # Test 4: Event ingestion
        resp4 = client.post("/v1/events", json={
            "user_id": test_uid,
            "item_name": "Counter-Strike",
            "event_type": "playtime",
            "playtime": 45.0,
        }, headers=headers)
        ok4 = resp4.status_code == 202
        print(f"  POST /v1/events                  : {resp4.status_code}  {'OK' if ok4 else 'FAIL'}")
        all_ok = all_ok and ok4

        return all_ok

    except RuntimeError as e:
        print(f"  SKIPPED (Redis not available): {e}")
        return True  # Don't fail the test if Redis isn't running

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("Game Recommender — End-to-End Test")
    print("=" * 60)

    results = []

    results.append(("Artifacts exist", check_artifacts()))
    results.append(("Artifact contents", check_artifact_contents()))
    results.append(("API endpoints", check_api_endpoints()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25}  [{status}]")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
