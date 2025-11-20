"""
FastAPI Application
Main API endpoints and application setup
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import hashlib
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
import logging
import os
from typing import Optional, Dict, Any
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

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper functions for HTML preview storage in Redis
def store_html_preview(preview_id: str, html_content: str, filename: str, ttl: int = 3600) -> bool:
    """Store HTML preview in Redis with TTL"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis not available, HTML preview will not be stored")
            return False
        
        preview_data = {
            'html': html_content,
            'timestamp': time.time(),
            'filename': filename
        }
        
        key = f"html_preview:{preview_id}"
        redis_client.setex(key, ttl, json.dumps(preview_data))
        logger.debug("Stored HTML preview %s in Redis with TTL %d", preview_id, ttl)
        return True
    except Exception as e:
        logger.error("Error storing HTML preview in Redis: %s", e, exc_info=True)
        return False

def get_html_preview_data(preview_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve HTML preview from Redis"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return None
        
        key = f"html_preview:{preview_id}"
        cached_data = redis_client.get(key)
        
        if cached_data:
            preview_data = json.loads(cached_data)
            logger.debug("Retrieved HTML preview %s from Redis", preview_id)
            return preview_data
        return None
    except Exception as e:
        logger.error("Error retrieving HTML preview from Redis: %s", e, exc_info=True)
        return None

def delete_html_preview(preview_id: str) -> bool:
    """Delete HTML preview from Redis"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        key = f"html_preview:{preview_id}"
        redis_client.delete(key)
        logger.debug("Deleted HTML preview %s from Redis", preview_id)
        return True
    except Exception as e:
        logger.error("Error deleting HTML preview from Redis: %s", e, exc_info=True)
        return False
from app.processor import get_processor
from app.universal_processor import get_universal_processor
from app.cache_manager import get_cache_manager

# HTML previews are now stored in Redis with 1-hour TTL
# Helper functions for Redis-based preview storage


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


@app.post(
    "/process",
    response_model=TaskResponse,
    tags=["Processing"],
    responses={
        200: {
            "description": "Processing results",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TaskResponse"},
                    "example": {
                        "task_id": "universal",
                        "status": "SUCCESS",
                        "message": "Processing completed",
                        "result": {
                            "input_type": "html",
                            "output_content": "<html>...</html>",
                            "corrections_count": 1
                        }
                    }
                },
                "text/html": {
                    "schema": {"type": "string"},
                    "example": "<html><body><p>This is a <u>test</u> sentence.</p></body></html>"
                }
            }
        },
        400: {"description": "Bad request - invalid file type or format parameter"},
        500: {"description": "Internal server error"}
    },
    summary="Process file for grammar correction",
    description="""
    Process uploaded file for grammar correction with support for multiple response formats.
    
    **Response Formats:**
    - `format=json` (default): Returns JSON with processing results
    - `format=html`: Returns HTML directly with `Content-Type: text/html` (HTML input only)
    
    **Supported Input Types:**
    - Images: .jpg, .jpeg, .png
    - HTML: .html, .htm
    - Archives: .zip (containing images/HTML)
    
    **HTML Response:**
    When `format=html` is used with HTML input, the response contains the corrected HTML
    with `<u>` tags wrapping corrected words. The response has `Content-Type: text/html`
    and can be rendered directly in a browser.
    
    **Preview:**
    For HTML responses, a `preview_id` is included in the response headers. Use
    `/process/preview/{preview_id}` to retrieve the HTML later.
    """
)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_processing: bool = Form(default=True),
    format: Optional[str] = Query(default="json", description="Response format: 'json' or 'html' (for HTML input only)")
):
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
        
        # Store filename before reading (file object may become unavailable)
        original_filename = file.filename
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Save uploaded file
        file_path = save_uploaded_file(file_content, original_filename)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files, "/tmp/uploads", 3600)
        background_tasks.add_task(cleanup_old_files, "/tmp/outputs", 3600)
        
        # Use universal processor for all input types
        logger.info("Processing %s with universal processor (async_processing=%s ignored)", original_filename, async_processing)
        
        universal_processor = get_universal_processor()
        result = universal_processor.process_any_input(file_path, output_dir="/tmp/outputs")
        
        # Add performance stats to response
        stats = universal_processor.get_performance_stats()
        result['performance_stats'] = stats
        
        # If format is html and input is HTML, return HTML directly (not escaped in JSON)
        if format and format.lower() == "html":
            # Check if this is HTML input
            input_type = result.get('input_type')
            if input_type == 'html' and result.get('success'):
                output_content = result.get('output_content')
                if output_content and isinstance(output_content, str):
                    # Generate preview ID for later retrieval
                    preview_id = str(uuid.uuid4())
                    
                    # Store HTML preview in Redis with 1-hour TTL
                    store_html_preview(preview_id, output_content, original_filename, ttl=3600)
                    
                    # Return as HTML with proper content type and preview ID header
                    response = HTMLResponse(
                        content=output_content,
                        status_code=200,
                        media_type="text/html"
                    )
                    response.headers["X-Preview-ID"] = preview_id
                    return response
            else:
                # If format=html but input is not HTML, return error
                raise HTTPException(
                    status_code=400,
                    detail=f"format=html is only available for HTML input files. Current input type: {input_type}"
                )
        
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


@app.get(
    "/process/preview/{preview_id}",
    response_class=HTMLResponse,
    tags=["Processing"],
    responses={
        200: {
            "description": "HTML preview of processed file",
            "content": {
                "text/html": {
                    "schema": {"type": "string"},
                    "example": "<html><body><p>This is a <u>test</u> sentence.</p></body></html>"
                }
            }
        },
        404: {"description": "Preview not found or expired"}
    },
    summary="Get HTML preview of processed file",
    description="""
    Retrieve the HTML preview of a processed file using the preview ID.
    
    Preview IDs are returned in the `X-Preview-ID` header when using `format=html`
    with the `/process` endpoint. Previews are stored for 1 hour.
    
    This endpoint is useful for:
    - Opening HTML previews in a browser
    - Sharing processed HTML results
    - Testing HTML rendering without re-processing
    
    **Example:**
    ```bash
    # Process file and get preview ID
    curl -X POST "http://localhost:8000/process?format=html" -F "file=@example.html"
    # Response includes: X-Preview-ID: abc123-def456-...
    
    # Retrieve preview
    curl "http://localhost:8000/process/preview/abc123-def456-..."
    ```
    """
)
async def get_html_preview(preview_id: str):
    """
    Get HTML preview of processed file by preview ID
    
    - **preview_id**: Preview ID from X-Preview-ID header (returned when format=html)
    """
    # Retrieve preview from Redis
    preview_data = get_html_preview_data(preview_id)
    
    if not preview_data:
        raise HTTPException(
            status_code=404,
            detail="HTML preview not found or expired. Previews are stored for 1 hour."
        )
    
    html_content = preview_data.get('html')
    if not html_content:
        raise HTTPException(
            status_code=404,
            detail="HTML preview data is invalid."
        )
    
    return HTMLResponse(
        content=html_content,
        status_code=200,
        media_type="text/html",
        headers={
            "X-Original-Filename": preview_data.get('filename', 'unknown')
        }
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
