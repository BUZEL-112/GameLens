# GameLens

A two-tower retrieval recommender for Steam games. The system mirrors how a production recommendation pipeline is structured end to end: a training pipeline that produces versioned artifacts, a FastAPI serving layer with cache-first lookups and a documented cold-start path, an online feature-refresh loop, and a retraining DAG with deployment gates and A/B routing. This is a personal project trained on the public UCSD Steam dataset; the demo path runs on a 10% sample for fast iteration.

---

## Architecture

```
gamelens/
├── configs/
│   └── config.yaml              single source of truth for hyperparameters, paths, thresholds
├── core_ml/
│   └── features.py              build_user_vector -- shared by training and serving, no I/O, no TF
├── training/
│   ├── pipeline.py              stages 1-6: clean, feature engineer, split, train, evaluate, export
│   ├── train.py                 two-tower model definition and in-batch InfoNCE training loop
│   └── evaluate.py              writes Recall@20 / NDCG@20 + baseline check into metrics.json
├── artifacts/
│   ├── u_tower.keras            user tower weights
│   ├── i_tower.keras            item tower weights
│   ├── item_embeddings.npy      precomputed item embedding matrix
│   ├── item_index.faiss         flat FAISS index over item embeddings
│   └── artifacts.pkl            serialized artifact bundle consumed by init_redis and the API
├── scripts/
│   └── init_redis.py            one-time bulk load of artifacts.pkl into Redis
├── recommendation_api/
│   └── services/
│       ├── retrieval.py         3-tier embedding lookup (cache -> tower -> cold-start fallback)
│       ├── reranking.py         filter played, genre diversity cap, context boost
│       └── nearline.py          daemon thread consuming events:stream, refreshes user embeddings
├── pipelines/
│   ├── orchestrator.py          Prefect DAG: retention, validation, retrain, eval, deploy, A/B
│   └── ab_testing.py            experiment routing and metric recording (flat JSON state)
├── gamelens-web/                Next.js frontend, proxies /rec/* to the API
├── tests/
│   └── test_feature_parity.py   pins build_user_vector against a frozen reference implementation
└── e2e_test.py                  artifact integrity checks + live-API smoke test via TestClient
```

The two towers are trained jointly with in-batch InfoNCE, so any user/item embedding dot product is directly usable as a ranking score. Item embeddings are precomputed once per training run and written to a FAISS flat index; user embeddings are computed on demand and cached in Redis.

---

## Design Decisions

**One feature function, two call sites, one test pinning them together.**
`core_ml/features.py` has no I/O and no TensorFlow dependency. `build_user_vector()` is called from `training/pipeline.py` (Stage 3, batch, full corpus) and from `recommendation_api/services/nearline.py` (online, one user at a time, per event). This is the actual fix for training-serving skew. `tests/test_feature_parity.py` freezes a reference implementation and asserts the shared function still matches it, so a refactor that quietly changes the math fails a test instead of silently degrading live recommendations.
Tradeoff accepted: the online path cannot apply corpus-level normalization (the last two vector dimensions are scaled against the global max across all users), because at serving time only one user's data is available. The nearline updater skips that step and documents why in its docstring.

**Redis is gated behind a readiness sentinel, not a connection check.**
The API's startup sequence checks for a `system:ready` key written at the end of `populate_from_artifacts()`. If it is missing, the app refuses to start with an explicit pointer to `python -m scripts.init_redis`, rather than booting and serving empty recommendations until someone notices. The same key backs Docker Compose's `depends_on: condition: service_healthy` for the frontend -- there is no window where the UI is live against a half-populated backend.

**Retrieval fails soft in three tiers, not with a 500.**
`RetrievalService.get_user_embedding()` checks a cached embedding in Redis, falls back to a stored feature vector run through the User Tower on a cache miss, and returns `None` only on a genuine cold start. Cold start is not an error: the API responds with a popularity-ranked list (`source: "popularity_fallback"`) and the frontend renders nothing rather than an error state, since a missing recommendation row is a UX non-event.

**The retraining DAG has gates, not just steps.**
`pipelines/orchestrator.py` runs Retention -> Validation -> Retraining -> Evaluation -> Deployment -> A/B Routing as a Prefect flow. The deployment task checks that the new model beats a popularity baseline and clears absolute Recall@20 / NDCG@20 thresholds from `config.yaml` (`training/evaluate.py` writes both checks into `metrics.json`). A model that overfits without beating "recommend the top 50 games" never reaches production, and the thresholds live in config rather than code.

**The chronological ordering assumption is documented, not hidden.**
`max_interactions_per_user` uses `.groupby("user_id").tail(N)`, treating row order as a proxy for chronological order because the public dataset carries no timestamps. This assumption is called out in `CONTRIBUTING.md`, in the function docstring, and in `training_manifest.json`'s `sampling.note` field -- because it is invisible until someone retrains on a differently-ordered dump and gets a confusing metric shift.

**Two onboarding paths share one artifact contract.**
`make setup-api` downloads artifacts trained on a 10% sample (explicitly labeled "not for production" in both the Makefile and the docs), putting a backend or frontend engineer against a real API in under five minutes without a GPU. `make setup-ml` runs the full pipeline. Both produce an `artifacts.pkl` with the identical schema, so nothing downstream needs to know which path ran.

---

## Quickstart

The fast path uses pre-built demo artifacts and skips training entirely.

```bash
git clone <repo> && cd gamelens
pip install -r requirements.txt
make setup-api          # downloads demo artifacts, starts Redis (~5 min)
make test               # parity + smoke tests against the live API
make setup-frontend     # boots the full stack with the Next.js UI
```

To run the full ML pipeline instead (downloads ~1 GB of raw data, trains on CPU in roughly 30-45 min):

```bash
make setup-ml
```

Environment variables, the Redis key schema, and stage-by-stage pipeline internals are in `CONTRIBUTING.md`.

---

## Project Structure

```
configs/config.yaml      single source of truth for hyperparameters, paths, thresholds
core_ml/                 feature logic shared between training and serving (no I/O, no TF)
training/                pipeline stages 1-6, model definitions, train.py, evaluate.py
recommendation_api/      FastAPI app -- routers, retrieval/reranking/nearline services, Redis
pipelines/               Prefect DAG -- retention, validation, retrain, A/B routing
scripts/init_redis.py    one-time bulk load of artifacts.pkl into Redis
gamelens-web/            Next.js frontend, proxies /rec/* to the API
tests/                   parity test pinning core_ml.features against a frozen reference
e2e_test.py              artifact integrity check + live-API smoke test
```

---

## Stack

Model: TensorFlow / Keras, FAISS | Serving: FastAPI, Pydantic v2, Redis | Orchestration: Prefect | Frontend: Next.js, Tailwind CSS | Infrastructure: Docker Compose

---

## Testing

- `tests/test_feature_parity.py` -- freezes the feature math and asserts the shared `build_user_vector` function still matches it. This is the test that matters: it catches the one failure mode that would silently degrade live recommendations without surfacing any other error.
- `e2e_test.py` -- checks embedding shapes and L2 norms in the saved artifacts, then runs a live smoke test against the FastAPI app via `TestClient`. API checks are skipped gracefully if Redis is not running so they do not block a pure offline run.

---

## Known Limitations

- Demo artifacts are a 10% sample -- recommendation quality at that scale is a structural-validity check, not a relevance benchmark.
- `_InMemoryRateLimiter` (`core/security.py`) is a single-process sliding window; a multi-worker deployment needs the Redis `INCR`+`EXPIRE` swap noted in the same file.
- A/B experiment state lives in flat JSON files (`pipelines/ab_testing.py`) -- safe on one machine, not safe under concurrent writers.
- No authentication on the frontend; user identity is a client-generated UUID in `localStorage`.

---

## Possible Next Steps

- Move A/B experiment state from JSON files to Redis so `record_metric` is safe under concurrent writers.
- Add a sequence-aware candidate generator alongside the two-tower model and route a traffic slice through the existing experiment framework.
- Replace the `.tail(N)` chronological proxy with real timestamps if a richer Steam dataset becomes available.