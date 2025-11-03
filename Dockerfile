FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for ML libraries and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tokenizers and sentencepiece compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Create virtual environment to avoid pip warnings
RUN python -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Skip Git LFS during build (prevents LFS download failures)
ENV GIT_LFS_SKIP_SMUDGE=1

# Upgrade pip and install Python dependencies
# Install essential build tools first
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel cython

# Install numpy first (critical, and some packages depend on it)
RUN /app/venv/bin/pip install --no-cache-dir "numpy==1.26.4"

# Install Pydantic FIRST (FastAPI depends on it)
RUN /app/venv/bin/pip install --no-cache-dir \
    "pydantic>=2.5.0,<3.0.0" \
    "pydantic-settings>=2.1.0,<3.0.0"

# Install PyTorch CPU version (large package, install early)
# Using 2.2.0 for better compatibility with transformers
RUN /app/venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0 \
    torchvision==0.17.0

# Install web framework dependencies (FastAPI needs Pydantic already installed)
RUN /app/venv/bin/pip install --no-cache-dir \
    "fastapi>=0.104.0,<1.0.0" \
    "uvicorn[standard]>=0.24.0,<1.0.0" \
    "python-multipart>=0.0.6,<1.0.0"

# Install async and HTTP clients
RUN /app/venv/bin/pip install --no-cache-dir \
    "aiofiles>=23.2.0,<24.0.0" \
    "aiohttp>=3.9.0,<4.0.0" \
    "requests>=2.31.0,<3.0.0" \
    "httpx>=0.25.0,<1.0.0"

# Install HTML processing (lxml can be problematic, install separately)
RUN /app/venv/bin/pip install --no-cache-dir "beautifulsoup4>=4.12.0,<5.0.0" && \
    /app/venv/bin/pip install --no-cache-dir "lxml>=4.9.0,<5.0.0"

# Install Redis (simple packages)
RUN /app/venv/bin/pip install --no-cache-dir \
    "redis>=4.6.0,<5.0.0" \
    "fakeredis>=2.32.0,<3.0.0"

# Install Celery and Flower (can be memory intensive)
RUN /app/venv/bin/pip install --no-cache-dir "celery[redis]>=5.3.0,<6.0.0" && \
    /app/venv/bin/pip install --no-cache-dir "flower>=2.0.0,<3.0.0"

# Install ML/transformers dependencies (after PyTorch and numpy)
# Pin transformers to compatible version with PyTorch 2.2.0
RUN /app/venv/bin/pip install --no-cache-dir \
    "tokenizers>=0.13.0,<1.0.0" \
    "sentencepiece>=0.2.0,<1.0.0" \
    "safetensors>=0.3.0,<1.0.0" \
    "huggingface-hub>=0.16.0,<1.0.0" \
    "transformers>=4.37.0,<4.50.0" \
    "accelerate>=0.20.0,<1.0.0" \
    "pillow>=9.5.0,<11.0.0" \
    "opencv-python-headless>=4.8.0,<5.0.0" \
    "easyocr>=1.7.0,<2.0.0"

# Install remaining utilities
RUN /app/venv/bin/pip install --no-cache-dir \
    "python-dotenv>=1.0.0,<2.0.0" \
    "prometheus-client>=0.19.0,<1.0.0" \
    "protobuf>=3.20.0,<5.0.0"

# Copy application code
COPY . .

# Download model from Hugging Face during deployment (as root, before switching user)
# This solves Git LFS issues by downloading model at build time instead of from Git
# MODEL_ID should be set as environment variable in Render Dashboard
# Render passes environment variables to Docker build, so we use ARG to capture them
ARG MODEL_ID
ARG HF_TOKEN
RUN mkdir -p ./model && \
    echo "=== Model Download Debug ===" && \
    echo "MODEL_ID ARG value: '${MODEL_ID:-not_set}'" && \
    echo "HF_TOKEN ARG value: '${HF_TOKEN:-not_set}'" && \
    /app/venv/bin/python -c "import os; model_id = '${MODEL_ID}' if '${MODEL_ID}' else None; token = '${HF_TOKEN}' if '${HF_TOKEN}' else None; print(f'Using MODEL_ID: {model_id}'); print(f'MODEL_ID is None: {model_id is None}'); print(f'MODEL_ID is empty: {model_id == \"\"}'); model_id = model_id.strip() if model_id and model_id != 'not_set' else None; from huggingface_hub import snapshot_download; (print(f'Downloading model: {model_id}') or snapshot_download(repo_id=model_id, local_dir='./model', token=token)) if model_id and model_id != 'not_set' else print('WARNING: MODEL_ID not set - model will not be downloaded. Set MODEL_ID as environment variable in Render Dashboard.')" && \
    echo "=== Model Download Complete ===" && \
    ls -lh ./model/ 2>/dev/null || echo "Note: Model directory listing unavailable"

# Create non-root user for runtime security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/uploads /tmp/cache /tmp/outputs && \
    chmod -R 755 /tmp && \
    chown -R appuser:appuser /tmp/uploads /tmp/cache /tmp/outputs || true

# Switch to non-root user for security (after all build steps)
USER appuser

# Expose port (supports PORT env var for Cloud Run, defaults to 8000)
EXPOSE 8000
ENV PORT=8000

# Health check with extended start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application (using virtual environment)
CMD ["sh", "-c", "/app/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]