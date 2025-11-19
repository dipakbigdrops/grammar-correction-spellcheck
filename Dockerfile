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
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
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

# Copy requirements and constraints first for better Docker layer caching
COPY requirements.txt constraints.txt ./

# Skip Git LFS during build (prevents LFS download failures)
ENV GIT_LFS_SKIP_SMUDGE=1

# Upgrade pip and install Python dependencies
# Install essential build tools first
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel cython

# Install numpy first (critical, and some packages depend on it)
# Use constraints to prevent any upgrades
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "numpy==1.26.4"

# Force numpy version before any other ML packages
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4"

# Install Pydantic FIRST (FastAPI depends on it)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "pydantic>=2.5.0,<3.0.0" \
    "pydantic-settings>=2.1.0,<3.0.0"

# Install PyTorch dependencies first (required when using --no-deps)
# Install pillow and requests BEFORE torchvision (torchvision requires them)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "typing-extensions>=4.8.0" \
    "filelock>=3.9.0" \
    "networkx>=2.6.0" \
    "sympy>=1.12.0" \
    "jinja2>=3.1.2" \
    "fsspec>=2023.6.0" \
    "packaging>=21.3" \
    "pillow==10.4.0" \
    "requests>=2.31.0,<3.0.0"

# Install PyTorch CPU version (large package, install early)
# Using 2.2.0 for better compatibility with transformers
# Using --no-deps to prevent numpy version conflicts
RUN /app/venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --no-deps

# Force numpy version after PyTorch installation
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4"

# Install web framework dependencies (FastAPI needs Pydantic already installed)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "fastapi>=0.104.0,<1.0.0" \
    "uvicorn[standard]>=0.24.0,<1.0.0" \
    "python-multipart>=0.0.6,<1.0.0"

# Install async and HTTP clients
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "aiofiles>=23.2.0,<24.0.0" \
    "aiohttp>=3.9.0,<4.0.0" \
    "requests>=2.31.0,<3.0.0" \
    "httpx>=0.25.0,<1.0.0"

# Install HTML processing (lxml can be problematic, install separately)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "beautifulsoup4>=4.12.0,<5.0.0" && \
    /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "lxml>=4.9.0,<5.0.0"

# Install Redis (simple packages)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "redis>=4.6.0,<5.0.0" \
    "fakeredis>=2.32.0,<3.0.0"

# Install Celery and Flower (can be memory intensive)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "celery[redis]>=5.3.0,<6.0.0" && \
    /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "flower>=2.0.0,<3.0.0"

# Install ML/transformers dependencies (after PyTorch and numpy)
# Pin transformers to compatible version with PyTorch 2.2.0
# Use constraints to prevent numpy/opencv upgrades
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "tokenizers>=0.13.0,<1.0.0" \
    "sentencepiece>=0.2.0,<1.0.0" \
    "safetensors>=0.3.0,<1.0.0" \
    "huggingface-hub>=0.16.0,<1.0.0" \
    "transformers>=4.37.0,<4.50.0" \
    "accelerate>=0.20.0,<1.0.0" \
    "pillow==10.4.0" \
    "opencv-python-headless==4.9.0.80" \
    "easyocr>=1.7.0,<2.0.0"

# Force opencv version after easyocr (easyocr may try to upgrade opencv)
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "opencv-python-headless==4.9.0.80"

# Force protobuf version after transformers (transformers may pull different version)
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "protobuf==4.25.3"

# Install remaining utilities
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "python-dotenv>=1.0.0,<2.0.0" \
    "prometheus-client>=0.19.0,<1.0.0"

# Final numpy, protobuf, and opencv version enforcement after all dependencies
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4" && \
    /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "protobuf==4.25.3" && \
    /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "opencv-python-headless==4.9.0.80"

# Verify numpy version is correct (fail build if wrong version)
RUN /app/venv/bin/python -c "import numpy; assert numpy.__version__ == '1.26.4', f'NumPy version mismatch! Expected 1.26.4, got {numpy.__version__}'; print(f'✓ NumPy version verified: {numpy.__version__}')" || exit 1

# Copy application code
COPY . .

# Download model from Hugging Face during deployment
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
# Verify NumPy version at runtime before starting
CMD ["sh", "-c", "/app/venv/bin/python -c \"import numpy; assert numpy.__version__.startswith('1.26'), f'CRITICAL: NumPy version {numpy.__version__} is incompatible! Expected 1.26.x'; print(f'✓ Runtime NumPy check: {numpy.__version__}')\" && /app/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]