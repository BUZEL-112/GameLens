.PHONY: help setup-api setup-frontend setup-ml test run-pipeline

# ── Default target ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "GameLens — Two-Tower Recommender System"
	@echo "======================================="
	@echo ""
	@echo "Choose a path based on your role:"
	@echo ""
	@echo "  make setup-api       Backend/API engineer  — 5 min, no GPU needed"
	@echo "  make setup-frontend  Frontend engineer     — 10 min, no GPU needed"
	@echo "  make setup-ml        ML engineer           — 30-45 min, full training"
	@echo ""
	@echo "Other targets:"
	@echo "  make test            Run the end-to-end test suite against a live API"
	@echo "  make run-pipeline    Execute the Prefect retraining DAG"
	@echo "  make redis-up        Start Redis only"
	@echo "  make redis-down      Stop all containers"
	@echo "  make init-redis      Populate Redis from model_artifacts/"
	@echo ""

# ── Fast path: API / Backend engineer ─────────────────────────────────────────
# Downloads pre-built artifacts and starts the full API stack in Docker.
# Target state: curl http://localhost:8000/health returns {"status":"ok"}
setup-api: _download-artifacts
	@echo "Starting Redis and API containers..."
	docker-compose up -d redis
	@echo "Waiting for Redis to be ready..."
	@sleep 3
	@echo "Populating Redis with artifacts..."
	python -m scripts.init_redis
	@echo ""
	@echo "API stack ready."
	@echo "  Health:  http://localhost:8000/health"
	@echo "  Docs:    http://localhost:8000/docs"
	@echo "  Run 'make test' to validate all endpoints."

# ── Fast path: Frontend engineer ──────────────────────────────────────────────
# Extends setup-api by also starting the Next.js development server.
# Target state: http://localhost:3000 is live and talking to the API.
setup-frontend: setup-api
	@echo "Starting GameLens web frontend..."
	cd gamelens-web && npm install && npm run dev

# ── Full path: ML / Training engineer ─────────────────────────────────────────
# Runs the full training pipeline: data download, cleaning, training, evaluation.
# Requires ~8GB RAM. Recommend a machine with a GPU for reasonable training time.
setup-ml:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "Starting full training pipeline..."
	@echo "  This will download ~1GB of raw data and train for 30-45 minutes."
	python -m training.train
	@echo ""
	@echo "Running offline evaluation..."
	python -m training.evaluate

# ── Shared utilities ──────────────────────────────────────────────────────────
redis-up:
	docker-compose up -d redis

redis-down:
	docker-compose down

init-redis:
	python -m scripts.init_redis

test:
	python e2e_test.py

run-pipeline:
	@echo "Running Prefect orchestrator..."
	python -m pipelines.orchestrator

# ── Internal helpers ──────────────────────────────────────────────────────────
# Downloads and extracts pre-built model artifacts from the GitHub release.
# These artifacts are sufficient to boot the API and return valid responses.
# They are NOT production-quality — they were trained on 10% of the dataset.
_download-artifacts:
	@if [ ! -d "model_artifacts" ] || [ -z "$$(ls -A model_artifacts 2>/dev/null)" ]; then \
		echo "Downloading pre-built model artifacts..."; \
		echo "  Source: GitHub Releases (demo artifacts — not for production use)"; \
		mkdir -p model_artifacts; \
		curl -L https://github.com/OWNER/REPO/releases/latest/download/model_artifacts.zip \
			-o model_artifacts.zip; \
		unzip -q model_artifacts.zip -d model_artifacts; \
		rm model_artifacts.zip; \
		echo "Artifacts extracted to model_artifacts/"; \
	else \
		echo "model_artifacts/ already populated — skipping download."; \
	fi
