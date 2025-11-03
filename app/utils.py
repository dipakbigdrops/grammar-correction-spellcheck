"""
Utility Functions
"""
import hashlib
import json
import logging
import os
import time
from typing import Optional

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# Redis connection
redis_client = None


def get_redis_client():
    """Get Redis client instance - uses fakeredis as fallback if Redis unavailable"""
    global redis_client
    if redis_client is None:
        # Skip Redis if host is explicitly set to skip or if host is localhost in production
        if settings.REDIS_HOST.lower() in ('skip', 'none', 'false', ''):
            logger.info("Redis disabled via configuration")
            redis_client = None
        else:
            try:
                # Try to connect to real Redis first
                logger.info(f"Attempting to connect to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=5,  # Increased timeout for Render
                    socket_timeout=5,  # Increased timeout for Render
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                redis_client.ping()
                logger.info(f"✅ Connected to Redis server successfully at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            except (redis.ConnectionError, redis.TimeoutError, OSError, Exception) as e:
                logger.warning(f"Real Redis not available at {settings.REDIS_HOST}:{settings.REDIS_PORT}: {e}")
                logger.info("Falling back to FakeRedis (in-memory) - cache will be in-memory only")
                try:
                    # Fallback to fakeredis (in-memory Redis for development)
                    import fakeredis  # pylint: disable=import-outside-toplevel
                    redis_client = fakeredis.FakeStrictRedis(decode_responses=True)
                    redis_client.ping()
                    logger.info("✅ Using FakeRedis (in-memory) - perfect for development!")
                except (ImportError, AttributeError) as fake_error:
                    logger.error("Failed to initialize FakeRedis: %s", fake_error)
                    redis_client = None
    return redis_client


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_cached_result(file_hash: str) -> Optional[dict]:
    """Get cached result from Redis"""
    if not settings.ENABLE_CACHING:
        return None
    
    client = get_redis_client()
    if client is None:
        return None
    
    try:
        cached = client.get(f"result:{file_hash}")
        if cached:
            logger.info("Cache hit for hash: %s", file_hash)
            return json.loads(cached)
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error("Error getting cached result: %s", e)

    return None


def set_cached_result(file_hash: str, result: dict):
    """Cache result in Redis"""
    if not settings.ENABLE_CACHING:
        return
    
    client = get_redis_client()
    if client is None:
        return
    
    try:
        client.setex(
            f"result:{file_hash}",
            settings.CACHE_TTL,
            json.dumps(result)
        )
        logger.info("Cached result for hash: %s", file_hash)
    except (redis.RedisError, TypeError) as e:
        logger.error("Error caching result: %s", e)


def create_directories():
    """Create necessary directories"""
    directories = [
        "/tmp/uploads",
        "/tmp/outputs",
        "/tmp/cache"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info("Created directory: %s", directory)
        except OSError as e:
            logger.error("Error creating directory %s: %s", directory, e)

def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "/tmp/uploads") -> str:
    """Save uploaded file and return path"""
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    base_name, ext = os.path.splitext(filename)
    
    # Limit base_name length to prevent filesystem issues
    max_base_length = 200  # Leave room for hash and extension
    if len(base_name) > max_base_length:
        base_name = base_name[:max_base_length]
    
    # Sanitize filename to remove problematic characters
    base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_', '.'))
    base_name = base_name.strip()
    
    # If base_name is empty after sanitization, use a default
    if not base_name:
        base_name = "file"
    
    unique_filename = f"{base_name}_{file_hash}{ext}"
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    logger.info("Saved uploaded file to: %s", file_path)
    return file_path


def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    """Clean up old files from directory"""
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    removed_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError as e:
                    logger.error("Error removing file %s: %s", file_path, e)

    if removed_count > 0:
        logger.info("Cleaned up %d old files from %s", removed_count, directory)
