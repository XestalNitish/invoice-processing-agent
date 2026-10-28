# Production Multi-Stage Dockerfile for Invoice Processing Agent
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install essential system libraries for OpenCV and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY Requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r Requirements.txt

# Copy application source code
COPY app/ /app/app/
COPY ui/ /app/ui/
COPY local_llm/ /app/local_llm/
COPY eval/ /app/eval/
COPY tests/ /app/tests/
COPY .env.example /app/.env

# Create persistent storage directories
RUN mkdir -p /app/storage/raw_images /app/storage/processed_images /app/exports

EXPOSE 8000 8501

# Default entrypoint runs FastAPI API service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
