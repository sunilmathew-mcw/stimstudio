# Multi-stage production container for StimStudio Headless API & Web Client
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for OpenCV, image decoding, and network healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install build system and Python dependencies
COPY pyproject.toml requirements*.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -e '.[server,cloud]'

# Copy application source tree and web client assets
COPY src/ ./src/
COPY README.md ./

# Re-install package in editable mode to register entrypoints and package data
RUN pip install --no-cache-dir -e '.[server]'

# Create persistent runtime data directories
RUN mkdir -p /app/data/task_images /app/data/experiments

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["python", "-m", "stimulus_forge.api.server", "--host", "0.0.0.0", "--port", "8000"]
