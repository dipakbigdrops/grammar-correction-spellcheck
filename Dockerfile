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

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Skip Git LFS during build (prevents LFS download failures)
ENV GIT_LFS_SKIP_SMUDGE=1

# Upgrade pip and install Python dependencies
# Install essential build tools first
RUN pip install --upgrade pip setuptools wheel cython

# Install numpy first (critical, and some packages depend on it)
RUN pip install --no-cache-dir "numpy==1.26.4"

# Install PyTorch CPU version first (large package, install early)
# This prevents dependency conflicts and reduces build time
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0 \
    torchvision==0.16.0

# Now install remaining dependencies from requirements.txt
# Split into two parts to avoid memory issues during build
RUN pip install --no-cache-dir \
    fastapi>=0.104.0,<1.0.0 \
    uvicorn[standard]>=0.24.0,<1.0.0 \
    python-multipart>=0.0.6,<1.0.0 \
    pydantic>=2.5.0,<3.0.0 \
    pydantic-settings>=2.1.0,<3.0.0 \
    aiofiles>=23.2.0,<24.0.0 \
    aiohttp>=3.9.0,<4.0.0 \
    requests>=2.31.0,<3.0.0 \
    httpx>=0.25.0,<1.0.0 \
    beautifulsoup4>=4.12.0,<5.0.0 \
    lxml>=4.9.0,<5.0.0 \
    redis>=4.6.0,<5.0.0 \
    celery[redis]>=5.3.0,<6.0.0 \
    fakeredis>=2.32.0,<3.0.0 \
    flower>=2.0.0,<3.0.0

# Install ML/transformers dependencies (after PyTorch and numpy)
RUN pip install --no-cache-dir \
    tokenizers>=0.13.0,<1.0.0 \
    sentencepiece>=0.2.0,<1.0.0 \
    safetensors>=0.3.0,<1.0.0 \
    huggingface-hub>=0.16.0,<1.0.0 \
    transformers>=4.30.0,<5.0.0 \
    accelerate>=0.20.0,<1.0.0 \
    pillow>=9.5.0,<11.0.0 \
    opencv-python-headless>=4.8.0,<5.0.0 \
    easyocr>=1.7.0,<2.0.0

# Install remaining utilities
RUN pip install --no-cache-dir \
    python-dotenv>=1.0.0,<2.0.0 \
    prometheus-client>=0.19.0,<1.0.0 \
    protobuf>=3.20.0,<5.0.0

# Copy application code
COPY . .

# Download model from Hugging Face during deployment
# This solves Git LFS issues by downloading model at build time instead of from Git
# MODEL_ID should be set as environment variable in Render Dashboard (available during build)
RUN mkdir -p ./model && \
    python -c "from huggingface_hub import snapshot_download; import os; model_id = os.environ.get('MODEL_ID'); token = os.environ.get('HF_TOKEN') or None; print(f'MODEL_ID from env: {model_id}'); snapshot_download(repo_id=model_id, local_dir='./model', token=token) if model_id else print('WARNING: MODEL_ID not set - model will not be downloaded. Set MODEL_ID as environment variable in Render Dashboard.')" && \
    echo "Model download process completed" && \
    ls -lh ./model/ 2>/dev/null || echo "Note: Model directory listing unavailable"

# Create necessary directories
RUN mkdir -p /tmp/uploads /tmp/cache /tmp/outputs && \
    chmod -R 755 /tmp

# Expose port (supports PORT env var for Cloud Run, defaults to 8000)
EXPOSE 8000
ENV PORT=8000

# Health check with extended start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]