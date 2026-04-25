FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY . .

# Copy model artifacts (mounted or copied at build time)
# In production, these would be fetched from cloud storage at startup.

EXPOSE 8000

CMD ["uvicorn", "recommendation_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
