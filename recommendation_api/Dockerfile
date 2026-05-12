# ---------- Builder stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Final stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages (no build tools)
COPY --from=builder /install /usr/local

# Copy only necessary app files
COPY recommendation_api ./recommendation_api

COPY training ./training
COPY configs ./configs
COPY data ./data
COPY model_artifacts ./model_artifacts

EXPOSE 8000

CMD ["uvicorn", "recommendation_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]