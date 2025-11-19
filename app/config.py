"""
Configuration Management
Handles all environment variables and settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Settings
    APP_NAME: str = "Grammar Correction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Redis Settings
    REDIS_HOST: str = "localhost"  # Use "redis" for production, "localhost" for local development
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""  # Set via environment variable in production
    
    # Celery Settings (auto-generated from Redis settings if not provided)
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 300  # 5 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = 240  # 4 minutes
    
    # Model Settings
    MODEL_PATH: str = "./model"
    MODEL_ID: str = ""  # Hugging Face model ID (e.g., "dipak-bigdrops/grammar-correction-model")
    HF_TOKEN: str = ""  # Hugging Face token (optional, only needed for private models)
    MODEL_MAX_LENGTH: int = 128
    MODEL_NUM_BEAMS: int = 5
    
    # OCR Settings
    OCR_LANGUAGES: list = ["en"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Processing Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png"]
    ALLOWED_HTML_EXTENSIONS: list = [".html", ".htm"]
    ALLOWED_ARCHIVE_EXTENSIONS: list = [".zip"]
    MAX_ZIP_EXTRACT_SIZE: int = 50 * 1024 * 1024  # 50MB total extracted
    MAX_FILES_IN_ZIP: int = 100  # Maximum files to process from ZIP
    CONTEXT_WORDS: int = 3
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour
    ENABLE_CACHING: bool = True  # Enabled - uses FakeRedis if real Redis unavailable
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    FLOWER_PORT: int = 5555
    FLOWER_USER: str = "admin"
    FLOWER_PASSWORD: str = ""
    
    # CORS Settings
    ALLOWED_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 200
    
    # Autoscaling Thresholds
    MIN_REPLICAS: int = 3
    MAX_REPLICAS: int = 100
    TARGET_CPU_UTILIZATION: int = 70
    TARGET_MEMORY_UTILIZATION: int = 80
    TARGET_REQUESTS_PER_SECOND: int = 100
    
    # Optimized Settings for High Throughput
    # Batch Processing Optimization
    BATCH_PROCESSING_TIMEOUT: int = 600  # 10 minutes for large batches
    ENABLE_BATCH_OPTIMIZATION: bool = True
    STREAMING_PROCESSING: bool = True  # Process files as they're extracted
    
    # Resource Optimization
    WORKER_CPU_LIMIT: float = 0.5  # 0.5 CPU per worker
    WORKER_MEMORY_LIMIT: int = 1024  # 1GB per worker
    WORKER_CONCURRENCY: int = 1  # One batch at a time
    
    # Multi-Level Caching
    CACHE_TTL_TEXT: int = 86400  # 24 hours for text content
    CACHE_TTL_MODEL: int = 3600  # 1 hour for model outputs
    CACHE_TTL_OCR: int = 7200  # 2 hours for OCR results
    CACHE_TTL_PARTIAL: int = 1800  # 30 minutes for partial results
    
    # Cache Hit Rate Optimization
    ENABLE_TEXT_CACHING: bool = True
    ENABLE_MODEL_CACHING: bool = True
    ENABLE_OCR_CACHING: bool = True
    ENABLE_PARTIAL_CACHING: bool = True
    
    # Performance Monitoring
    METRICS_INTERVAL: int = 60  # 1 minute
    CACHE_HIT_RATE_TRACKING: bool = True
    PERFORMANCE_TRACKING: bool = True
    
    # Context and Processing
    ENABLE_EARLY_TERMINATION: bool = True  # Skip processing if no text
    ENABLE_PARALLEL_OCR: bool = True  # Process multiple images simultaneously
    
    # Environment
    ENVIRONMENT: str = "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse list fields from JSON strings if they are strings
        # This handles cases where environment variables are set as JSON strings
        list_fields = [
            'ALLOWED_ORIGINS', 'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS',
            'OCR_LANGUAGES', 'ALLOWED_IMAGE_EXTENSIONS', 'ALLOWED_HTML_EXTENSIONS',
            'ALLOWED_ARCHIVE_EXTENSIONS'
        ]
        
        for field in list_fields:
            value = getattr(self, field, None)
            if isinstance(value, str):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        setattr(self, field, parsed)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, try splitting by comma
                    if value.strip() == "*":
                        setattr(self, field, ["*"])
                    else:
                        # Split by comma and strip whitespace
                        setattr(self, field, [item.strip() for item in value.split(",") if item.strip()])
        
        # Auto-generate Celery URLs if not provided
        if not self.CELERY_BROKER_URL or not self.CELERY_RESULT_BACKEND:
            redis_auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            redis_url = f"redis://{redis_auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            
            if not self.CELERY_BROKER_URL:
                self.CELERY_BROKER_URL = redis_url
            
            if not self.CELERY_RESULT_BACKEND:
                self.CELERY_RESULT_BACKEND = redis_url
        
        # Validate batch processing settings
        if self.MAX_FILES_IN_ZIP > 1000:
            self.MAX_FILES_IN_ZIP = 1000  # Cap at 1000 for performance
        
        if self.MAX_ZIP_EXTRACT_SIZE > 500 * 1024 * 1024:  # 500MB
            self.MAX_ZIP_EXTRACT_SIZE = 500 * 1024 * 1024  # Cap at 500MB


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create global settings instance
settings = get_settings()