"""
FastAPI Application
Main API endpoints and application setup
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
import logging
import os
from typing import Optional
import time
from contextlib import asynccontextmanager

# Import custom middleware
from app.middleware import RateLimitMiddleware, CircuitBreakerMiddleware, RequestTrackingMiddleware

from app.config import settings
from app.models import (
    TaskResponse, TaskStatusResponse, ProcessResult,
    HealthResponse, ErrorResponse, ProcessingStatus, InputType
)
from app.tasks import process_grammar_correction
from app.utils import (
    get_redis_client, compute_file_hash, get_cached_result,
    set_cached_result, save_uploaded_file, cleanup_old_files
)
from app.processor import get_processor
from app.universal_processor import get_universal_processor
from app.cache_manager import get_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    
    # Test Redis connection
    try:
        redis_client = get_redis_client()
        if redis_client:
            logger.info("Redis connected successfully")
        else:
            logger.warning("Redis connection failed - caching disabled")
    except (ConnectionError, OSError) as e:
        logger.warning("Redis connection test failed: %s", e)
    
    # Create necessary directories
    try:
        os.makedirs("/tmp/uploads", exist_ok=True)
        os.makedirs("/tmp/outputs", exist_ok=True)
        logger.info("Created necessary directories")
    except OSError as e:
        logger.error("Failed to create directories: %s", e)
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-performance Grammar Correction API with OCR and HTML support",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add custom middleware for production
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(CircuitBreakerMiddleware, failure_threshold=5, timeout=60)
app.add_middleware(RateLimitMiddleware, requests_per_minute=1000, burst=2000)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to {}".format(settings.APP_NAME),
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    redis_connected = False
    try:
        client = get_redis_client()
        if client:
            client.ping()
            redis_connected = True
    except (ConnectionError, AttributeError) as e:
        logger.debug("Redis health check failed: %s", e)
    
    # Check if processor can be initialized
    model_loaded = False
    ocr_available = False
    beautifulsoup_available = False
    image_reconstruction_available = False
    html_reconstruction_available = False
    
    try:
        # Check if model path exists and is valid
        model_loaded = os.path.exists(settings.MODEL_PATH) and os.path.exists(os.path.join(settings.MODEL_PATH, "config.json"))
        
        # Check OCR availability
        try:
            import easyocr
            ocr_available = True
        except ImportError:
            ocr_available = False
        except (OSError, RuntimeError):
            ocr_available = False

        # Check BeautifulSoup availability
        try:
            from bs4 import BeautifulSoup
            # Test with a simple HTML string
            soup = BeautifulSoup("<html><body>test</body></html>", 'html.parser')
            beautifulsoup_available = True
        except ImportError:
            beautifulsoup_available = False
        except (ValueError, AttributeError):
            beautifulsoup_available = False

        # Check image reconstruction capabilities
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2
            import numpy as np
            # Test basic image operations
            test_img = Image.new('RGB', (100, 100), color='white')
            test_array = np.array(test_img)
            image_reconstruction_available = True
        except ImportError:
            image_reconstruction_available = False
        except (OSError, ValueError):
            image_reconstruction_available = False

        # Check HTML reconstruction capabilities
        try:
            from bs4 import BeautifulSoup
            from difflib import Differ
            # Test HTML parsing and text extraction
            test_html = "<html><body><p>Test content</p></body></html>"
            soup = BeautifulSoup(test_html, 'html.parser')
            text = soup.get_text()
            differ = Differ()
            html_reconstruction_available = True
        except ImportError:
            html_reconstruction_available = False
        except (ValueError, AttributeError):
            html_reconstruction_available = False

    except (OSError, ImportError) as e:
        logger.debug("Model/OCR health check failed: %s", e)
    
    return HealthResponse(
        status="healthy" if redis_connected and model_loaded else "degraded",
        version=settings.APP_VERSION,
        redis_connected=redis_connected,
        grammar_model_loaded=model_loaded,
        ocr_available=ocr_available,
        beautifulsoup_available=beautifulsoup_available,
        image_reconstruction_available=image_reconstruction_available,
        html_reconstruction_available=html_reconstruction_available
    )


@app.post("/process", response_model=TaskResponse, tags=["Processing"])
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_processing: bool = Form(default=True)
):
    """
    Process uploaded file for grammar correction - Universal Input Support
    
    - **file**: Image (.jpg, .png, .jpeg), HTML (.html, .htm), or ZIP archive containing images/HTML
    - **async_processing**: Parameter kept for API compatibility but ignored (always processes synchronously)
    
    Returns immediate processing results with corrected text and reconstructed content
    Supports both single files and ZIP archives with batch processing optimization
    """
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        allowed_extensions = (
            settings.ALLOWED_IMAGE_EXTENSIONS + 
            settings.ALLOWED_HTML_EXTENSIONS + 
            settings.ALLOWED_ARCHIVE_EXTENSIONS
        )
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Save uploaded file
        file_path = save_uploaded_file(file_content, file.filename)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, "/tmp/uploads", 3600)
        background_tasks.add_task(cleanup_old_files, "/tmp/outputs", 3600)
        
        # Use universal processor for all input types
        logger.info("Processing %s with universal processor (async_processing=%s ignored)", file.filename, async_processing)
        
        universal_processor = get_universal_processor()
        result = universal_processor.process_any_input(file_path, output_dir="/tmp/outputs")
        
        # Add performance stats to response
        stats = universal_processor.get_performance_stats()
        result['performance_stats'] = stats
        
        return JSONResponse(content={
            "task_id": "universal",
            "status": "SUCCESS" if result.get('success') else "FAILURE",
            "message": "Processing completed with universal processor",
            "result": result,
            "estimated_completion_seconds": result.get('processing_time_seconds', 0)
        })
    
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Error processing file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """
    Get status of a processing task
    
    - **task_id**: Task ID returned from /process endpoint
    """
    try:
        # Handle special task IDs
        if task_id == "sync" or task_id == "cached":
            return TaskStatusResponse(
                task_id=task_id,
                status=ProcessingStatus.SUCCESS,
                progress=100,
                result={"message": "Task completed"}
            )
        
        # Try to get task result
        try:
            task_result = AsyncResult(task_id)
            
            if task_result.state == 'PENDING':
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=ProcessingStatus.PENDING,
                    progress=0
                )
            elif task_result.state == 'STARTED':
                progress = task_result.info.get('progress', 50) if isinstance(task_result.info, dict) else 50
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=ProcessingStatus.STARTED,
                    progress=progress
                )
            elif task_result.state == 'SUCCESS':
                result = task_result.result
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=ProcessingStatus.SUCCESS,
                    progress=100,
                    result=result
                )
            elif task_result.state == 'FAILURE':
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=ProcessingStatus.FAILURE,
                    progress=0,
                    error=str(task_result.info)
                )
            else:
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=ProcessingStatus.PENDING,
                    progress=0
                )
            
            return response
        except AttributeError:
            # Celery backend not configured - return pending status
            logger.warning("Celery backend not configured, returning pending status for task %s", task_id)
            return TaskStatusResponse(
                task_id=task_id,
                status=ProcessingStatus.PENDING,
                progress=0,
                result={"message": "Task status unavailable - Celery backend not configured"}
            )
    
    except (AttributeError, ValueError) as e:
        logger.error("Error getting task status: %s", e)
        # Return a valid response instead of raising exception
        return TaskStatusResponse(
            task_id=task_id,
            status=ProcessingStatus.PENDING,
            progress=0,
            result={"error": "Task status unavailable"}
        )


@app.get("/download/{filename}", tags=["Output"])
async def download_file(filename: str):
    """
    Download processed output file
    
    - **filename**: Name of the output file
    """
    file_path = os.path.join("/tmp/outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    try:
        # Get universal processor stats
        universal_processor = get_universal_processor()
        processor_stats = universal_processor.get_performance_stats()
        
        # Get cache stats
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_cache_stats()
        
        # Try to get Celery stats
        try:
            from app.celery_app import celery_app
            celery_stats = celery_app.control.inspect().stats()
            active_tasks = celery_app.control.inspect().active()
            
            return {
                "status": "operational",
                "processor_stats": processor_stats,
                "cache_stats": cache_stats,
                "celery_stats": {
                    "workers": celery_stats,
                    "active_tasks": active_tasks
                }
            }
        except (AttributeError, ConnectionError) as celery_error:
            logger.warning("Celery metrics unavailable: %s", celery_error)
            return {
                "status": "operational",
                "processor_stats": processor_stats,
                "cache_stats": cache_stats,
                "celery_stats": {
                    "workers": "unavailable",
                    "active_tasks": "unavailable",
                    "message": "Celery backend not configured"
                }
            }
    except (OSError, AttributeError) as e:
        logger.error("Error getting metrics: %s", e)
        return {
            "status": "degraded",
            "error": "Metrics unavailable"
        }


@app.get("/performance", tags=["Monitoring"])
async def performance_stats():
    """
    Get detailed performance statistics
    """
    try:
        universal_processor = get_universal_processor()
        cache_manager = get_cache_manager()
        
        return {
            "processor_stats": universal_processor.get_performance_stats(),
            "cache_stats": cache_manager.get_cache_stats(),
            "cache_size": cache_manager.get_cache_size(),
            "timestamp": time.time()
        }
    except (OSError, AttributeError) as e:
        logger.error("Error getting performance stats: %s", e)
        return {
            "error": "Performance stats unavailable",
            "timestamp": time.time()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG
    )
