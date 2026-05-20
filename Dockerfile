# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies into a virtualenv so we can easily copy them to the final image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install CPU-only torch to save space, then install other requirements
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Runner
# ==============================================================================
FROM python:3.11-slim AS runner

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

# Install runtime system dependencies (if any needed for FAISS/Chroma/OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY src/ /app/src/
COPY eval/ /app/eval/
COPY tests/ /app/tests/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/test_reports && \
    chmod 777 /app/data /app/logs /app/test_reports

# Expose FastAPI port
EXPOSE 8000

# Healthcheck to verify the API is running
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
