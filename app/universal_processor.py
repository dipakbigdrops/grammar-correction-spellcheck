"""
Universal Processor for All Input Types
Handles single images, HTML documents, and ZIP files efficiently
Optimized for 50K RPM under $100/month
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from app.config import settings as optimized_settings
from app.cache_manager import get_cache_manager
from app.processor import get_processor
from app.zip_handler import get_zip_handler

logger = logging.getLogger(__name__)


class UniversalProcessor:
    """Universal processor that handles all input types efficiently"""
    
    def __init__(self):
        self.processor = get_processor()
        self.zip_handler = get_zip_handler()
        self.cache_manager = get_cache_manager()
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'single_files': 0,
            'zip_files': 0,
            'cache_hits': 0,
            'processing_time': 0
        }
    
    def process_any_input(self, file_path: str, output_dir: str = "/tmp") -> Dict[str, Any]:
        """
        Process any input type (single file or ZIP) efficiently
        
        Args:
            file_path: Path to input file
            output_dir: Directory for output files
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Determine input type
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Check cache first (universal caching)
            file_hash = self._compute_file_hash(file_path)
            cached_result = None
            try:
                cached_result = self.cache_manager.get_file_cache(file_hash)
            except (KeyError, AttributeError, Exception) as e:
                # If cache lookup fails, log and continue without cache
                logger.warning("Cache lookup failed for %s: %s. Continuing without cache.", 
                             os.path.basename(file_path), e)
            
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info("Cache hit for %s", os.path.basename(file_path))
                return {
                    **cached_result,
                    'cached': True,
                    'processing_time_seconds': 0
                }
            
            # Process based on file type
            if file_extension in optimized_settings.ALLOWED_ARCHIVE_EXTENSIONS:
                result = self._process_zip_file(file_path, output_dir)
                self.stats['zip_files'] += 1
            else:
                result = self._process_single_file(file_path, output_dir)
                self.stats['single_files'] += 1
            
            # Update stats
            self.stats['total_processed'] += 1
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            result['processing_time_seconds'] = round(processing_time, 2)
            
            # Cache successful results
            if result.get('success'):
                self.cache_manager.set_file_cache(file_hash, result)
            
            return result
            
        except (OSError, IOError, RuntimeError) as e:
            logger.error("Error processing %s: %s", file_path, e, exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'input_type': 'unknown',
                'processing_time_seconds': time.time() - start_time
            }
    
    def _process_single_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Process single file (image or HTML)"""
        try:
            # Use existing processor for single files
            result = self.processor.process_input(file_path, output_dir)
            
            # Preserve the original input_type from processor (html, image, etc.)
            # Don't overwrite it with 'single_file' as we need to know the actual type
            # Add universal metadata without overwriting input_type
            if 'input_type' not in result:
                result['input_type'] = 'single_file'  # Fallback if not set
            result['file_count'] = 1
            result['batch_processing'] = False
            
            return result
            
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error processing single file %s: %s", file_path, e)
            return {
                'success': False,
                'error': str(e),
                'input_type': 'single_file',
                'file_count': 1,
                'batch_processing': False
            }
    
    def _process_zip_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Process ZIP file with optimized batch processing"""
        try:
            # Use ZIP handler for batch processing
            result = self.zip_handler.process_zip_file(file_path, self.processor, output_dir)
            
            # Add universal metadata
            result['input_type'] = 'zip_file'
            result['batch_processing'] = True
            
            # Optimize result structure for universal response
            if result.get('success'):
                # Flatten results for easier client consumption
                flattened_results = []
                for file_result in result.get('results', []):
                    flattened_results.append({
                        'filename': file_result.get('filename'),
                        'success': file_result.get('success'),
                        'original_text': file_result.get('original_text', ''),
                        'corrected_text': file_result.get('corrected_text', ''),
                        'corrections_count': file_result.get('corrections_count', 0),
                        'output_content': file_result.get('output_content'),
                        'processing_time_seconds': file_result.get('processing_time_seconds', 0)
                    })
                
                result['flattened_results'] = flattened_results
            
            return result
            
        except (OSError, IOError, RuntimeError) as e:
            logger.error("Error processing ZIP file %s: %s", file_path, e)
            return {
                'success': False,
                'error': str(e),
                'input_type': 'zip_file',
                'batch_processing': True
            }
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash for caching"""
        try:
            import hashlib  # pylint: disable=import-outside-toplevel
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (OSError, IOError) as e:
            logger.error("Error computing file hash: %s", e)
            return str(time.time())  # Fallback to timestamp
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_processing_time = (
            self.stats['processing_time'] / self.stats['total_processed']
            if self.stats['total_processed'] > 0 else 0
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['total_processed'] * 100
            if self.stats['total_processed'] > 0 else 0
        )
        
        return {
            'total_processed': self.stats['total_processed'],
            'single_files': self.stats['single_files'],
            'zip_files': self.stats['zip_files'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'avg_processing_time_seconds': round(avg_processing_time, 2),
            'total_processing_time_seconds': round(self.stats['processing_time'], 2)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_processed': 0,
            'single_files': 0,
            'zip_files': 0,
            'cache_hits': 0,
            'processing_time': 0
        }


# Global universal processor instance
_universal_processor = None

def get_universal_processor() -> UniversalProcessor:
    """Get or create global universal processor instance"""
    global _universal_processor
    if _universal_processor is None:
        _universal_processor = UniversalProcessor()
    return _universal_processor
