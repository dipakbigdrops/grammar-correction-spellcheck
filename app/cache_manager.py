"""
Multi-Level Cache Manager
Implements aggressive caching for 50K RPM under $100/month
"""
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List

import redis

from app.config import settings as optimized_settings
from app.utils import get_redis_client

logger = logging.getLogger(__name__)


class CacheManager:
    """Multi-level cache manager for aggressive optimization"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = {
            'text': optimized_settings.CACHE_TTL_TEXT,
            'model': optimized_settings.CACHE_TTL_MODEL,
            'ocr': optimized_settings.CACHE_TTL_OCR,
            'partial': optimized_settings.CACHE_TTL_PARTIAL,
            'result': optimized_settings.CACHE_TTL
        }
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'text_hits': 0,
            'model_hits': 0,
            'ocr_hits': 0,
            'partial_hits': 0,
            'result_hits': 0
        }
    
    def _get_cache(self, key: str, cache_type: str = 'result') -> Optional[Any]:
        """Get cached value with statistics tracking"""
        if not optimized_settings.ENABLE_CACHING or not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                self.cache_stats['hits'] += 1
                # Initialize cache_type_hits if it doesn't exist
                cache_type_key = f'{cache_type}_hits'
                if cache_type_key not in self.cache_stats:
                    self.cache_stats[cache_type_key] = 0
                self.cache_stats[cache_type_key] += 1
                logger.debug("Cache hit for %s", key)
                return json.loads(cached)
            self.cache_stats['misses'] += 1
            logger.debug("Cache miss for %s", key)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error("Error getting cache %s: %s", key, e)
            return None
    
    def _set_cache(self, key: str, value: Any, cache_type: str = 'result') -> bool:
        """Set cached value with TTL"""
        if not optimized_settings.ENABLE_CACHING or not self.redis_client:
            return False
        
        try:
            ttl = self.cache_ttl.get(cache_type, optimized_settings.CACHE_TTL)
            self.redis_client.setex(key, ttl, json.dumps(value))
            logger.debug("Cached %s with TTL %d", key, ttl)
            return True
        except (redis.RedisError, TypeError) as e:
            logger.error("Error setting cache %s: %s", key, e)
            return False
    
    def get_text_cache(self, text: str) -> Optional[Dict[str, Any]]:
        """Cache based on text content - highest hit rate expected"""
        if not optimized_settings.ENABLE_TEXT_CACHING:
            return None
        
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return self._get_cache(f"text:{text_hash}", 'text')
    
    def set_text_cache(self, text: str, result: Dict[str, Any]) -> bool:
        """Cache text processing result"""
        if not optimized_settings.ENABLE_TEXT_CACHING:
            return False
        
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return self._set_cache(f"text:{text_hash}", result, 'text')
    
    def get_model_cache(self, input_text: str) -> Optional[str]:
        """Cache model outputs - medium hit rate expected"""
        if not optimized_settings.ENABLE_MODEL_CACHING:
            return None
        
        input_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()
        return self._get_cache(f"model:{input_hash}", 'model')
    
    def set_model_cache(self, input_text: str, output_text: str) -> bool:
        """Cache model output"""
        if not optimized_settings.ENABLE_MODEL_CACHING:
            return False
        
        input_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()
        return self._set_cache(f"model:{input_hash}", output_text, 'model')
    
    def get_ocr_cache(self, image_path: str) -> Optional[List[Any]]:
        """Cache OCR results - medium hit rate expected"""
        if not optimized_settings.ENABLE_OCR_CACHING:
            return None
        
        try:
            from app.utils import compute_file_hash
            image_hash = compute_file_hash(image_path)
            return self._get_cache(f"ocr:{image_hash}", 'ocr')
        except (OSError, IOError) as e:
            logger.error("Error getting OCR cache for %s: %s", image_path, e)
            return None
    
    def set_ocr_cache(self, image_path: str, ocr_results: List[Any]) -> bool:
        """Cache OCR results"""
        if not optimized_settings.ENABLE_OCR_CACHING:
            return False
        
        try:
            from app.utils import compute_file_hash
            image_hash = compute_file_hash(image_path)
            return self._set_cache(f"ocr:{image_hash}", ocr_results, 'ocr')
        except (OSError, IOError) as e:
            logger.error("Error setting OCR cache for %s: %s", image_path, e)
            return False
    
    def get_partial_cache(self, partial_key: str) -> Optional[Dict[str, Any]]:
        """Cache partial results - low hit rate but high value"""
        if not optimized_settings.ENABLE_PARTIAL_CACHING:
            return None
        
        return self._get_cache(f"partial:{partial_key}", 'partial')
    
    def set_partial_cache(self, partial_key: str, result: Dict[str, Any]) -> bool:
        """Cache partial result"""
        if not optimized_settings.ENABLE_PARTIAL_CACHING:
            return False
        
        return self._set_cache(f"partial:{partial_key}", result, 'partial')
    
    def get_file_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached file result - original caching method"""
        return self._get_cache(f"result:{file_hash}", 'result')
    
    def set_file_cache(self, file_hash: str, result: Dict[str, Any]) -> bool:
        """Set cached file result - original caching method"""
        return self._set_cache(f"result:{file_hash}", result, 'result')
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': round(hit_rate, 2),
            'text_hits': self.cache_stats.get('text_hits', 0),
            'model_hits': self.cache_stats.get('model_hits', 0),
            'ocr_hits': self.cache_stats.get('ocr_hits', 0),
            'partial_hits': self.cache_stats.get('partial_hits', 0),
            'result_hits': self.cache_stats.get('result_hits', 0)
        }
    
    def clear_cache(self, cache_type: str = None) -> bool:
        """Clear cache - optionally by type"""
        if not self.redis_client:
            return False
        
        try:
            if cache_type:
                pattern = f"{cache_type}:*"
            else:
                pattern = "*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info("Cleared %d cache entries for pattern %s", len(keys), pattern)
            return True
        except redis.RedisError as e:
            logger.error("Error clearing cache: %s", e)
            return False
    
    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size by type"""
        if not self.redis_client:
            return {}
        
        try:
            sizes = {}
            for cache_type in ['text', 'model', 'ocr', 'partial', 'result']:
                keys = self.redis_client.keys(f"{cache_type}:*")
                sizes[cache_type] = len(keys)
            return sizes
        except redis.RedisError as e:
            logger.error("Error getting cache size: %s", e)
            return {}


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
