# GameLens — Two-Tower Game Recommender

A production-grade Two-Tower recommendation system for Steam games. Built with FastAPI, Redis, FAISS, and TensorFlow — packaged for local development, Docker deployment, and scheduled retraining via Prefect.

---

## Who are you?

Choose your path. Each one gets you to a working environment in the shortest time possible.

| Role | Time | What you need |
|---|---|---|
| [Backend / API Engineer](#backend--api-engineer) | ~5 min | Running API + all endpoints |
| [Frontend Engineer](#frontend-engineer) | ~10 min | API + GameLens UI at localhost:3000 |
| [ML / Training Engineer](#ml--training-engineer) | ~45 min | Full training pipeline + evaluation |
| [DevOps / Infrastructure](#devops--infrastructure) | — | Docker, volumes, health checks |

---

## Backend / API Engineer

**You will end this section with:** a running API at `http://localhost:8000` returning real recommendation data, having never touched a training script.

### Prerequisites
- Docker and Docker Compose
- Python 3.11+

### Steps

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2. Start Redis and populate it with pre-built artifacts**
```bash
make setup-api
```

This command downloads pre-built model artifacts (~200 MB), starts Redis, and populates it with embeddings and similarity tables. The artifacts were trained on a 10% sample of the Steam dataset — they return structurally valid, reasonable recommendations and are sufficient for all API development work.

> **Not for production.** The dummy artifacts are clearly labelled. Run `make setup-ml` to train a full model when you need it.

**3. Confirm it's working**
```bash
# Health check
curl http://localhost:8000/health

# Personalized recommendations
curl -H "X-API-Key: dev-insecure-key" \
  "http://localhost:8000/v1/recommendations?user_id=76561197970982479&count=5"

# Similar items
curl -H "X-API-Key: dev-insecure-key" \
  "http://localhost:8000/v1/items/Counter-Strike/similar?count=5"
```

**4. Explore the full API schema**

Visit **http://localhost:8000/docs** — FastAPI generates interactive documentation from the Pydantic schemas automatically. Every endpoint, request body, and response shape is documented there.

**5. Run the end-to-end test suite**
```bash
make test
```

---

## Frontend Engineer

**You will end this section with:** the GameLens UI running at `http://localhost:3000`, with recommendations and similar-game navigation backed by a live API.

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+

### Steps

```bash
make setup-frontend
```

This runs `make setup-api` (API stack) and then starts the Next.js development server in `gamelens-web/`. Hot-reload is active — changes to the frontend are reflected instantly.

- **API:** http://localhost:8000
- **UI:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs — use this to understand exact JSON shapes before writing components

### Key response shapes

```jsonc
// GET /v1/recommendations
{
  "recommendations": [
    { "item_name": "Counter-Strike", "score": 0.94, "reason": "Action", "boosted": false }
  ],
  "source": "two_tower",
  "model_version": "v20260515_120000",
  "latency_ms": 12.4
}

// GET /v1/items/{item_name}/similar
{
  "item_name": "Counter-Strike",
  "similar_items": [
    { "item_name": "Counter-Strike: Global Offensive", "score": 0.97 }
  ]
}
```

All endpoints require the `X-API-Key: dev-insecure-key` header (set via `API_KEY` env var).

---

## ML / Training Engineer

**You will end this section with:** trained model artifacts in `model_artifacts/`, offline evaluation metrics printed to stdout, and a Prefect-orchestrated retraining pipeline you can trigger on demand.

### Prerequisites
- Python 3.11+
- ~8 GB RAM
- GPU recommended (CPU training works, takes longer)

### Steps

**1. Run the full training pipeline**
```bash
make setup-ml
```

This downloads ~1 GB of raw Steam data, runs cleaning and feature engineering, trains the Two-Tower model (~30 minutes on CPU), and evaluates it.

To run stages individually:
```bash
# Download + train only (skip evaluation)
python -m training.train

# Skip download if data already exists
python -m training.train --skip-download

# Disable 10% sampling — use full dataset (much slower)
python -m training.train --disable-sampling

# Evaluate an existing model
python -m training.evaluate
```

**2. Understand what was trained**

After training, `model_artifacts/manifests/training_manifest.json` contains:
- The version tag and timestamp
- Sampling parameters (seed, interaction window, sample fraction)
- Row counts at each stage (raw → tail(N) → sample → train/test split)
- MD5 checksums of every artifact file

**3. Run the full retraining DAG**
```bash
make run-pipeline
```

This executes the Prefect orchestrator: `Retention → Validation → Retraining → Evaluation → Deployment → A/B Routing`. See `pipelines/orchestrator.py` for the task dependency graph.

**4. Training configuration**

All hyperparameters live in `configs/config.yaml`. Key training settings:

```yaml
training:
  random_seed: 42                    # Controls all randomness globally
  max_interactions_per_user: 20      # Time-aware window (latest N interactions)
  sample_fraction: 0.1               # 10% sample for development
  embedding_dim: 128
  epochs: 100
  batch_size: 512
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for a full explanation of the training pipeline stages.

---

## DevOps / Infrastructure

### Docker Compose Services

```
redis     — Redis 7 (LRU eviction, 2 GB cap)
api       — FastAPI + uvicorn (volume-mounted model_artifacts)
web       — Next.js frontend (depends: api service_healthy)
```

### Volume Mounts

`model_artifacts/` is **never baked into the API image** — it is always mounted as a read-only volume at runtime:

```yaml
volumes:
  - ./model_artifacts:/app/model_artifacts:ro
```

If the volume is missing at startup, the API fails fast with a clear error message rather than serving stale baked artifacts.

### Health Checks

The API fails to start if Redis has not been populated (missing `system:ready` sentinel key). Run `python -m scripts.init_redis` before starting the API container.

The `web` service uses `depends_on: api: condition: service_healthy`, so it will not start until the API is passing health checks.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database |
| `REDIS_PASSWORD` | _(empty)_ | Redis password |
| `ARTIFACTS_PATH` | `model_artifacts` | Path to artifact directory |
| `API_KEY` | `dev-insecure-key` | API authentication key |

Set `API_KEY` to a strong secret in production.

### Build Context Optimization

A `.dockerignore` file at the repository root excludes `data/`, `model_artifacts/`, `.git/`, `node_modules/`, and `__pycache__/` from the Docker build context. This keeps build times fast regardless of local artifact or dataset sizes.

---

## Architecture

```
User Request
     |
     v
[FastAPI Gateway]  ← API key middleware on all /v1/ routes
     |
     +-- GET /v1/recommendations
     |        |
     |        v
     |   [RetrievalService]  -->  FAISS ANN Index  -->  Top-K candidates
     |        |
     |        v
     |   [ReRankingService]  -->  Business rules, diversity, freshness
     |        |
     |        v
     |   Personalized response
     |
     +-- POST /v1/events
     |        |
     |        v
     |   [Redis Feature Store]  -->  events:stream
     |        |
     |        v
     |   [NearlineUpdater]  -->  Refreshes user embeddings in background
     |
     +-- GET /health
```

**Two-Tower model:**
- **User Tower:** 322-dim behavioral vector → 128-dim embedding
- **Item Tower:** item ID embedding (32-dim) + genre/tag content (320-dim) → 128-dim embedding
- **Training:** InfoNCE loss, in-batch negatives, hard negative mining, cosine LR decay
- **Serving:** FAISS approximate nearest neighbor over pre-computed item embeddings

---

## Project Structure

```
project_009/
|
+-- configs/config.yaml          <- all settings (hyperparameters, paths, thresholds)
|
+-- training/                    <- ML pipeline (data → model → artifacts)
|   +-- train.py                 <- entry point: python -m training.train
|   +-- evaluate.py              <- entry point: python -m training.evaluate
|   +-- pipeline.py              <- Stages 1-6 (feature extraction, training loop)
|   +-- models.py                <- User Tower / Item Tower definitions
|
+-- core_ml/                     <- shared feature logic (training + serving)
|   +-- features.py              <- safe_parse, flatten_to_string
|
+-- recommendation_api/          <- FastAPI serving layer
|   +-- main.py                  <- app entry point, lifespan, middleware
|   +-- routers/                 <- route handlers (events, recommendations)
|   +-- services/                <- retrieval, reranking, nearline updater
|   +-- core/                    <- feature store, config, security
|   +-- models/schemas.py        <- Pydantic request/response schemas
|   +-- Dockerfile
|
+-- pipelines/                   <- scheduled pipeline jobs
|   +-- orchestrator.py          <- Prefect DAG (run with: make run-pipeline)
|   +-- retrain.py               <- weekly retraining job
|   +-- data_retention.py        <- daily archival + user snapshots
|   +-- data_validation.py       <- manifest-based input validation
|   +-- ab_testing.py            <- experiment management
|
+-- scripts/
|   +-- init_redis.py            <- populates Redis from model_artifacts/
|
+-- gamelens-web/                <- Next.js frontend
+-- model_artifacts/             <- output of training (gitignored, volume-mounted)
+-- docker-compose.yml
+-- Makefile                     <- role-based entry points (make help)
+-- e2e_test.py
```

---

## All Makefile Targets

```bash
make help            # Print this role guide
make setup-api       # Backend: download artifacts + start API stack
make setup-frontend  # Frontend: API stack + Next.js dev server
make setup-ml        # ML: full training pipeline
make test            # Run e2e_test.py against a live API
make run-pipeline    # Execute the full Prefect retraining DAG
make redis-up        # Start Redis only
make redis-down      # Stop all containers
make init-redis      # Populate Redis from model_artifacts/
```
