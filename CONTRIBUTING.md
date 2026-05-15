# Contributing to GameLens

This document covers the details that belong behind the README's front door — environment setup, codebase internals, and how to work on each layer of the system.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Configuration](#project-configuration)
- [Training Pipeline Internals](#training-pipeline-internals)
- [API Internals](#api-internals)
- [Redis and the Feature Store](#redis-and-the-feature-store)
- [Running Evaluations](#running-evaluations)
- [Retraining Pipeline (Prefect DAG)](#retraining-pipeline-prefect-dag)
- [A/B Testing](#ab-testing)
- [Code Style](#code-style)

---

## Environment Setup

### Python

This project targets Python 3.11. Use a virtual environment to avoid dependency conflicts.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Redis

The API requires a running Redis instance. The easiest way is Docker:

```bash
docker-compose up -d redis
```

Or install Redis locally and run `redis-server`. The default config expects `localhost:6379`.

### Environment Variables

The API reads config from environment variables (with sensible defaults for local dev):

| Variable | Default | Notes |
|---|---|---|
| `REDIS_HOST` | `localhost` | Use `redis` inside Docker Compose |
| `REDIS_PORT` | `6379` | |
| `REDIS_DB` | `0` | |
| `REDIS_PASSWORD` | _(empty)_ | Set for production |
| `ARTIFACTS_PATH` | `model_artifacts` | Path read by the API on startup |
| `API_KEY` | `dev-insecure-key` | Required header: `X-API-Key` |

For local development, defaults work without any `.env` file. For Docker, values are set in `docker-compose.yml`.

---

## Project Configuration

All settings are centralized in `configs/config.yaml`. The file is organized into sections:

```yaml
data_ingestion:    # Download URLs for raw Steam datasets
data_cleaning:     # Paths for raw and processed data directories
feature_engineering:  # Input/output paths for the transformation step
training:          # Hyperparameters, sampling settings, random seed
evaluation:        # Metric thresholds and deployment gates
orchestration:     # Manifest directory for inter-task contracts
serving:           # Redis connection, artifact path
pipelines:         # Retention schedule, archive dirs
```

**Key training settings to understand:**

```yaml
training:
  random_seed: 42                   # Passed to Python random, NumPy, and TensorFlow
  max_interactions_per_user: 20     # Rolling window — keeps most recent N interactions
  sample_fraction: 0.1              # 10% random sample after windowing (dev mode only)
```

The `random_seed` is set globally at the top of `train.py::main()` before any data loading, so it controls weight initialization, data shuffling, negative sampling, and train/test splits uniformly.

The `max_interactions_per_user` window uses `.groupby("user_id").tail(N)` on the interaction dataframe. This relies on the Steam dataset's inherent row ordering as a chronological proxy — later rows represent more recently added games. This assumption is documented in `training_manifest.json` under `sampling.note`.

---

## Training Pipeline Internals

The full training pipeline lives in `training/`. Entry point: `python -m training.train`.

### Stage-by-Stage Breakdown

| Stage | Function | What it does |
|---|---|---|
| 0 | `LoadDataService.run()` | Downloads raw `.json.gz` files from UCSD |
| 0 | `CleanDataService.run()` | Merges user interactions with game metadata |
| 0 | `FeatureEngineeringService.run()` | Applies rolling window, sampling, train/test split |
| 1 | `stage1_split_sides()` | Separates item catalog from interaction log |
| 2 | `stage2_process_items()` | Builds genre/tag vocabulary and content matrix |
| 3 | `stage3_process_users()` | Aggregates weighted user history into fixed-dim vectors |
| 4 | `stage4_build_training_pairs()` | Samples positive interactions, mines hard negatives |
| 5 | `stage5_assemble_tensors()` | Converts Python data structures to TF tensors |
| 6 | `stage6_train_loop()` | Gradient descent with InfoNCE loss and cosine LR |
| — | Embedding extraction | Runs the item tower over all items in batches |
| — | FAISS index | Builds inner-product ANN index |
| — | Similarity table | Pre-computes top-10 similar items for every game |

### Artifact outputs

After `train.py` completes, `model_artifacts/` contains:

```
u_tower.keras              User Tower saved model
i_tower.keras              Item Tower saved model
u_tower.weights.h5         User Tower weights
i_tower.weights.h5         Item Tower weights
item_embeddings.npy        (N_items, 128) float32 array
item_index.faiss           FAISS flat inner-product index
artifacts.pkl              Vocabularies, user vectors, dimension metadata
similarity_table.pkl       Dict[item_name, List[{item_name, score}]]
training_history.pkl       Per-epoch loss history
manifests/
  training_manifest.json   Version tag, sampling params, row counts, MD5 checksums
```

### Shared Feature Logic

`core_ml/features.py` contains `safe_parse()` and `flatten_to_string()`. These functions are used by both `training/feature_engineering.py` and `recommendation_api/services/nearline.py` — this is intentional. Any change to how features are constructed must go through `core_ml/` to maintain training-serving parity.

---

## API Internals

Entry point: `recommendation_api/main.py`

### Startup Sequence

1. The lifespan function runs before the app accepts any traffic.
2. It checks for a `system:ready` sentinel key in Redis. If missing, the API raises a `RuntimeError` with instructions and refuses to start.
3. It checks that `artifacts.pkl` exists at `ARTIFACTS_PATH`. If missing, it raises a `RuntimeError` with the volume mount message.
4. It instantiates `RetrievalService`, `ReRankingService`, and `NearlineUpdater`.
5. It starts the `NearlineUpdater` background thread, which consumes from `events:stream`.
6. The app yields (starts serving traffic).

### Middleware

`APIKeyMiddleware` (in `core/security.py`) is applied at the app level. It reads the `X-API-Key` header and compares it against `settings.api_key`. Requests to `/health` and `/docs` bypass authentication. All `/v1/*` routes require the key.

### Route Registration Order

`/items/search` is registered **before** `/items/{item_name}/similar` in `routers/recommendations.py`. This is intentional — FastAPI matches routes in declaration order, and a dynamic segment like `{item_name}` would capture the literal string `"search"` if declared first.

---

## Redis and the Feature Store

`recommendation_api/core/feature_store.py` wraps all Redis operations.

### Key Schema

| Key pattern | Type | Contents |
|---|---|---|
| `user:{user_id}:vec` | Hash | User embedding vector (serialized float32) |
| `item:{item_name}:meta` | Hash | Genres, tags, item_id |
| `item:{item_name}:similar` | String | JSON list of similar items |
| `popular` | String | JSON list of top-50 popular items |
| `events:stream` | Stream | Raw interaction events from POST /v1/events |
| `system:ready` | String | Sentinel — JSON with model_version and populated_at |

### Populating Redis

Run `python -m scripts.init_redis` after Redis is running. This script:

1. Loads artifacts from `model_artifacts/`
2. Calls `FeatureStore.populate_from_artifacts()` using a Redis pipeline for efficiency
3. Sets the `system:ready` sentinel key

The API will not start without this sentinel.

---

## Running Evaluations

```bash
python -m training.evaluate
```

The evaluation suite (`training/evaluate.py`) runs against the test split of the training data. It produces:

- **Recall@K** and **NDCG@K** for K ∈ {10, 20, 50, 100}
- **Popularity baseline comparison** — the model must beat a popularity-ranked list
- **Embedding sanity checks** — cosine similarity distributions and cluster coherence

Results are printed to stdout and written to `model_artifacts/manifests/metrics.json`. This file is consumed by the Prefect orchestrator's deployment gate, which compares results against `evaluation.min_recall_20` and `evaluation.min_ndcg_20` in `config.yaml`.

---

## Retraining Pipeline (Prefect DAG)

```bash
make run-pipeline
```

This runs the full Prefect flow defined in `pipelines/orchestrator.py`. The DAG:

```
Data Retention
    |
Data Validation   ← reads retention_manifest.json, verifies archive Parquet
    |
Model Retraining  ← runs training/train.py, writes training_manifest.json
    |
Offline Evaluation ← runs training/evaluate.py, writes metrics.json
    |
Conditional Deployment Gate
    |-- FAIL: Halt. Prefect marks run as Failed.
    |-- PASS: Push artifacts to Redis, set system:ready sentinel.
         |
         A/B Traffic Routing ← concludes old experiment, starts new 10% split
```

Each task communicates with the next via a JSON manifest written to `model_artifacts/manifests/`. This makes each stage independently testable and observable.

### Manifests

| File | Written by | Read by |
|---|---|---|
| `retention_manifest.json` | `data_retention.py` | `data_validation.py` |
| `validation_manifest.json` | `data_validation.py` | `orchestrator.py` (logged only) |
| `training_manifest.json` | `training/train.py` | `orchestrator.py` (deployment task) |
| `metrics.json` | `training/evaluate.py` | `orchestrator.py` (gate check) |

---

## A/B Testing

`pipelines/ab_testing.py` provides `ABTestingManager`, which manages experiments stored in Redis.

```bash
# List current experiments
python -m pipelines.ab_testing --action status

# Create a new experiment routing 10% to a new model version
python -m pipelines.ab_testing --action create \
  --name "v2_test" \
  --control v20260501_120000 \
  --treatment v20260515_083000 \
  --traffic 0.1

# Analyze results
python -m pipelines.ab_testing --action analyze --name "v2_test"

# Conclude and record winner
python -m pipelines.ab_testing --action conclude --name "v2_test" --winner treatment
```

When the Prefect pipeline's deployment task succeeds, it automatically calls `ABTestingManager` to conclude the previous experiment and create a new one routing 10% of traffic to the newly deployed model.

**A/B testing validity note:** For valid comparisons, both the control and treatment models should be trained from the same `random_seed`. Since `random_seed` is now centralized in `config.yaml` and applied globally in `train.py`, all models trained with the same config are directly comparable. If you change the seed, document it in the experiment metadata.

---

## Code Style

- Python 3.11 type hints throughout
- Pydantic v2 for all schema definitions
- `from __future__ import annotations` at the top of every module
- Logging via `training/utils/logger.py` (the same logger instance used across training)
- All custom exceptions via `training/utils/exception.py`
- No inline magic numbers — all constants in `config.yaml`
