FROM python:3.11-slim

# ─────────────────────────────────────────────────────────────────────────────
# Multimodal Clinical AI Copilot — HuggingFace Spaces Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/openi/images data/openi/reports data/literature data/pdfs

# HuggingFace Spaces runs as user 1000
RUN chown -R 1000:1000 /app
USER 1000

# Set python path so fastmcp can find mcp_server module
ENV PYTHONPATH=/app

# Render.com injects a $PORT env var at runtime — expose it
EXPOSE 8000

# Use shell form so $PORT is evaluated at runtime by Render
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
