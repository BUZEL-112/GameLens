# GameLens: Building an End-to-End Two-Tower Recommendation Pipeline for Steam Games

Recommender systems are arguably the most influential applications of machine learning in production today, impacting more daily users than almost any other class of model. Whether navigating Amazon, Netflix, Spotify, or massive online marketplaces, these systems operate continuously in the background to combat information overload. By analyzing behavior to surface relevant and diverse content, they solve a critical challenge for modern platforms: turning an overwhelming catalog into a curated, frictionless user experience that drives engagement.

In a real-world production environment, the operational impact of a recommender system depends far more on its surrounding infrastructure than on the model alone. A precise model is useless if it cannot serve predictions reliably or scale under production load.

This post provides a walkthrough of how I built GameLens, an end-to-end Two-Tower recommendation pipeline for Steam games. Rather than limiting scope to the math inside an offline notebook, this breakdown focuses on the backend infrastructure required to operationalize and serve predictions in a live environment, bridging the gap between model development and low-latency production inference.

---

## Data Collection

The dataset is the public [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data), specifically two files:

- **`australian_users_items.json.gz`** -- user interaction records. Each record contains a user ID and a nested list of games the user has played, along with playtime in minutes.

```json
{
  "user_id": "76561197970982479",
  "items_count": 277,
  "items": [
    {
      "item_id": "10",
      "item_name": "Counter-Strike",
      "playtime_forever": 320,
      "playtime_2weeks": 0
    },
    {
      "item_id": "20",
      "item_name": "Team Fortress Classic",
      "playtime_forever": 15,
      "playtime_2weeks": 0
    }
  ]
}
```

- **`steam_games.json.gz`** -- game metadata. Each record contains a game ID, title, genres, and tags.

```json
{
  "id": "730",
  "app_name": "Counter-Strike: Global Offensive",
  "title": "Counter-Strike: Global Offensive",
  "genres": ["Action", "Free to Play"],
  "tags": ["FPS", "Multiplayer", "Shooter", "Competitive", "Tactical"]
}
```

The raw dataset covers roughly 11,000 games and 880,000 Australian Steam users.

---

## Data Preparation

Before the towers can learn anything, the raw Steam dumps need to be turned into a clean interaction table with stable item metadata.

The data cleaning process handles this in three steps. First, it streams the user interactions file line by line and flattens the nested structure: each line is one user record containing a list of game objects, and the output is one row per `(user, game)` pair with user ID, item ID, total playtime, and item name. The file is streamed rather than loaded in full because the dataset is large enough to be painful in memory on a developer machine. Second, it parses the game metadata file to extract ID, genres, tags, and title. Third, it performs a left join on item ID, keeping all user interactions even when game metadata is missing, and produces a merged interaction dataset.

This merged dataset then passes through feature engineering, which applies a configurable data sampling cut, enforces a maximum interaction cap per user (so power users do not dominate the training signal), and creates separate train and test sets. The test split is a stratified hold-out, not a random row split, so every test user has at least a minimum number of known interactions to evaluate against.

Playtime is converted to an implicit engagement score before anything else touches it. The raw value is log-transformed via `log1p` to dampen outliers (a user with 10,000 hours should not be weighted 100x more than one with 100 hours), then normalized per-user to a 0--5 range so that active and casual users sit on a comparable scale. This engagement score is what the training pipeline operates on, never raw playtime minutes.

---

## Training

The training pipeline turns the cleaned interaction table into two embedding spaces that can be compared with a single dot product. It runs six stages in sequence.

### Stage 1: User/Item Splitting

The merged dataframe is separated into two independent frames: an item catalog of unique games with their genres and tags, and a user interaction table of user IDs, item names, and interaction scores. If the interaction score was not computed during preparation, this stage calculates it inline using the same `log1p` normalization described above. 

### Stage 2: Item Processing

The item-side vocabulary and feature matrix are built. The top 20 genres and top 300 tags by frequency are selected, and each game is encoded as a 320-dimensional multi-hot vector concatenating genre flags and tag flags. This is the item tower's content input at training time and the feature source for precomputing item embeddings.

### Stage 3: User Processing

User feature vectors are constructed for each user, the feature computation logic produces a 322-dimensional vector: the first 320 dimensions are an interaction-weighted average of the item content vectors for every game the user has played; dimension 321 is normalized total playtime; dimension 322 is normalized number of distinct games played. This logic is shared with the online serving layer -- it is the exact same process at training time and at nearline update time, preventing training-serving skew by design rather than by convention.

### Stage 4: Pair Construction

Training pairs are assembled. For each positive interaction (a game the user actually played), the pipeline samples two hard negatives and two random negatives. Hard negatives are games that are content-similar to the user's play history but not in it -- found by computing cosine similarity between all catalog items and the user's content centroid, then sampling from the top-50 unplayed results. Random negatives provide a contrast baseline. The final dataset is shuffled before training.

### Stage 5: Tensor Assembly

All training data is pre-assembled into contiguous NumPy arrays. User feature vectors and item content vectors are looked up once and stacked, so the training loop avoids dictionary lookups per batch.

### Stage 6: Training Loop

Both towers are shallow DNNs (input -> BatchNorm -> Dense(256) -> Dropout(0.2) -> Dense(128) -> Dense(128) -> L2 Norm) that end in L2 normalization, so their outputs are unit vectors. The similarity between a user embedding and an item embedding is the dot product of two unit vectors, which is exactly cosine similarity. InfoNCE is applied in-batch: for a batch of size 512, the loss is the cross-entropy of a 512x512 similarity matrix where the diagonal represents the true (user, item) pairs. Every off-diagonal item in the same batch acts as a negative for each user. The learning rate follows a cosine decay schedule from 1e-3 to 1e-5.

After training, the pipeline saves the following artifacts for deployment:

- `u_tower.keras` and `i_tower.keras` -- the trained Keras models
- `item_index.faiss` -- a `IndexFlatIP` (inner product) FAISS index over all precomputed item embeddings
- `artifacts.pkl` -- vocabularies, user feature vectors, item content matrix, and metadata
- `item_embeddings.npy` -- raw item embedding matrix for re-indexing after retraining
- `similarity_table.pkl` -- precomputed item-to-item nearest neighbor table for fast similar-item lookups
- `training_manifest.json` -- paths and a version tag for every artifact, used by the deployment step

---

## Evaluation

Evaluation is where the system proves that the model is doing more than rediscovering the most popular games.

The evaluation process runs several phases after loading the saved artifacts:

**Accuracy metrics.** A held-out set of users (default 10% of users with at least a minimum number of interactions) is embedded through the User Tower. For each test user, the model retrieves its top-K items via FAISS and the result is compared against the user's held-out interactions. Recall@K measures how many of a user's actual games appear in the top K recommendations. NDCG@K weights hits by their rank position so that finding the right game at rank 1 counts more than finding it at rank 20. Both are computed at K = 5, 10, and 20.

**Popularity baseline.** A non-personalized recommender that always returns the globally most-played 20 games is evaluated on the same test users with the same metric. This baseline is the floor: if the two-tower model cannot beat it, it is not providing personalization and should not be deployed. The evaluation logs a "lift over baseline" number, which is the ratio of the model's Recall@20 to the baseline's Recall@20.

**Embedding sanity checks.** The evaluation script verifies that all item embeddings are L2-normalized (a failure here would break the inner product search), checks for dead dimensions (dimensions with zero or near-zero variance across the embedding space are a sign of collapsed training), and verifies that genre-based clusters are geometrically separable -- games that share a genre should have higher pairwise cosine similarity than random pairs.

**Semantic consistency.** Two additional checks validate that the nearest-neighbor structure reflects real game relationships rather than spurious training artifacts. Tag Jaccard similarity measures how much tag overlap the recommended games have with the query game's tags. Co-play consistency checks whether the model's nearest neighbors for a given game are games that users historically played together. Both should be higher for model recommendations than for a random selection.

Finally, the evaluation produces a set of metrics that serve as a deployment gate, which is helpful in the retraining process. The model does not go live unless it beats the popularity baseline and exceeds minimum configurable thresholds for recall and NDCG. These thresholds are defined externally in a configuration file, allowing them to be adjusted without modifying the core logic. 

---

## Inference

At serving time, the expensive work has already been done. Inference is mostly a Redis cache lookup, a FAISS nearest-neighbor search, and a reranking pass. The API is built with FastAPI and ensures it is fully initialized before accepting traffic, preventing the system from silently serving empty recommendations.

The service exposes the following main endpoints:
- `GET /v1/recommendations`: Provides personalized recommendations for a specific user (the prediction endpoint).
- `GET /v1/items/{item_name}/similar`: Provides item-to-item recommendations based on a query game.
- `POST /v1/events`: Ingests real-time user interactions to update features in the background.

### Service Initialization

Before the API can serve any traffic, it must execute a strict initialization sequence to load the artifacts produced during training and prime the feature store. The service refuses to accept requests until this process completes successfully, preventing it from silently serving empty or fallback recommendations in a half-initialized state.

1. **Model and Index Loading:** The service pulls the latest deployment manifest (generated post-evaluation). It loads the trained user tower into memory and mounts the precomputed FAISS approximate nearest neighbor index.
2. **Artifact Hydration:** Static artifacts, such as the item vocabularies and the precomputed item-to-item similarity table, are loaded into local application memory for fast access.
3. **Feature Store Priming:** The system checks the Redis feature store for a "ready" sentinel signal. If absent, it halts and requires an initialization process to populate Redis with the baseline popularity list, the latest user feature vectors, and precomputed user embeddings.
4. **Traffic Readiness:** Only after all models are loaded, FAISS indices are mounted, and Redis is verified, does the service flag itself as ready and begin routing live traffic to the endpoints.

### Processing a Recommendation Request

When a client calls `GET /v1/recommendations`, the request follows a tiered lookup:

1. **Redis cache hit.** The service checks Redis for a precomputed user embedding. If found, it is used immediately, completely bypassing the model.
2. **On-the-fly inference.** If no cached embedding exists, the service retrieves the user's stored feature vector from Redis and runs a single forward pass through the user tower. The resulting embedding is written back to Redis for subsequent requests.
3. **Cold-start fallback.** If the user is entirely new and has no stored features, the service falls back to a precomputed global popularity list stored in Redis. The response explicitly indicates this was a fallback so the caller can distinguish it from a personalized result.
4. **Candidate retrieval.** For known users, the embedding queries the FAISS index, which returns the top candidates by inner product score.
5. **Reranking.** Retrieved candidates pass through a `ReRankingService` that applies three business rules in order: filter already-played items, enforce genre diversity (configurable max items per genre), and apply contextual score boosts (e.g., a 5% boost for Action/Indie games when the context is `cart`).

### Processing a Similar-Items Request

The `GET /v1/items/{item_name}/similar` endpoint bypasses the user embedding entirely:

1. **Item Lookup:** The service retrieves the precomputed embedding for the requested game from Redis.
2. **Nearest Neighbor Search:** It queries the FAISS index using this item embedding to find the most conceptually similar games. (In practice, a precomputed similarity table generated during training often acts as a fast-path cache for this endpoint).
3. **Contextual Re-ranking:** Just like personalized recommendations, these candidates pass through the final re-ranking phase to apply necessary contextual boosts and enforce genre diversity before being returned to the client.

### Nearline Feature Updates

The `POST /v1/events` endpoint handles continuous ingestion of interaction data. Events are accepted with a 202 status and processed asynchronously:

1. **Synchronous played-set update.** The user's played-items set in Redis is updated immediately, so the reranking filter reflects the new interaction right away.
2. **Async stream push.** The event is pushed onto a Redis Stream (`events:stream`) for background processing.
3. **Background consumption.** The `NearlineUpdater` daemon polls the stream every 5 seconds in batches of 50, identifies affected users, and for each one rebuilds their feature vector using the same `build_user_vector()` function from `core_ml/features.py` that training uses.
4. **Cache refresh.** The updated feature vector is run through the user tower and the resulting embedding overwrites the cached value in Redis. Subsequent recommendation requests immediately use the fresh embedding.

---

## Orchestration and Retraining

Retraining is treated as a candidate-generation step, not an automatic promotion step. The entire lifecycle is managed by a Prefect-based orchestration pipeline that defines a DAG of six tasks, ensuring that a weak or broken model never reaches production.

The pipeline is fueled by real-time data: as the live inference API serves recommendations, it continuously ingests user interaction events into a Redis Stream. These events are eventually gathered and processed by the orchestrator.

The automated pipeline executes in the following sequence:

1. **Data Retention and Archival.** The pipeline connects to the live feature store and archives all recent interaction events to permanent disk storage. It captures a snapshot of current user profiles for auditing and trims the live event stream to free memory.

2. **Data Validation.** Before any expensive compute is spent on training, the pipeline validates the newly archived data: schema consistency, corruption checks, and row count verification. If the data is broken, the pipeline halts immediately.

3. **Model Retraining.** The full dual-tower training pipeline runs from scratch. It computes engagement scores with the standard log-transformation, generates new item and user embeddings, precomputes the item-to-item similarity table, and persists all artifacts under a unique timestamped version tag. This step does *not* touch the live feature store.

4. **Offline Evaluation.** The newly trained model is evaluated against the held-out test users. It must produce accuracy metrics (Recall and NDCG at multiple K values) and beat the popularity baseline to prove it learned meaningful patterns.

5. **Conditional Deployment.** This is the primary safety gate. The task loads the metrics manifest and checks two conditions: the model must have passed the baseline gate, and its Recall@20 and NDCG@20 must exceed the minimum thresholds defined in `config.yaml`. If either check fails, the task raises an error and the pipeline terminates -- the old model continues serving undisturbed. If both pass, the task pushes the new embeddings, vocabularies, and user vectors into Redis and updates the sentinel flag, promoting the new model to production.

6. **A/B Traffic Routing.** Rather than instantly shifting 100% of users to the new model, the pipeline concludes any prior experiment and configures a new A/B test that routes 10% of live traffic to the newly deployed model for real-world performance monitoring.

---

## Deployment

The deployment architecture is entirely containerized with Docker Compose and built around one strict rule: the API must never serve from an uninitialized or half-populated feature store.

The system defines three services that start up in a precise dependency chain:

### Redis (Feature Store and Message Broker)

Redis acts as the central data layer for the entire application. It serves four roles:

- **Low-latency cache.** Precomputed embeddings, similarity tables, and the popularity fallback list are stored in Redis so the API can access them in sub-millisecond time.
- **Real-time message broker.** It hosts the `events:stream` Redis Stream. When users interact with the frontend, the API pushes events directly into this stream for background processing.
- **Deployment gate.** It holds the `system:ready` sentinel key and the active model version tag, controlling whether the API will accept traffic.
- **Persistent state.** Redis is configured with a persistent volume. If the container restarts, it retains all embeddings and avoids requiring a full re-hydration.

A 2 GB memory cap with `allkeys-lru` eviction policy keeps resource usage bounded.

### The Recommendation API

The FastAPI service handles all inference logic. It depends on Redis being available and runs a continuous health check. On startup, it checks for the `system:ready` sentinel key. If the key is missing, the service refuses to start with a clear error message explaining how to run the initialization script. This prevents silent failures and empty recommendations.

### The Web Frontend

The web UI is the final service in the chain. It is configured with a `service_healthy` dependency condition on the API container, meaning it will not start until the API has passed its health check. This guarantees that the user interface never loads against an unavailable backend.

### The Startup Contract

Due populating the Redis feature store with millions of embeddings from cold storage takes time and can fail. The api startup requires starting the Redis cache first, manually hydrating it from the precomputed model artifacts, and explicitly writing the "ready" sentinel flag. Only then can the API and Web services be started, at which point they will read the sentinel flag, verify the system is fully populated, and safely begin serving live traffic.


