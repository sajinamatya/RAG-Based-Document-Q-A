import json
import hashlib
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
import redis

from config.settings import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis caching for RAG system to cache search results and embeddings."""
    
    def __init__(self):
        self.settings = settings
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            self.client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from arguments."""
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        if not self.is_connected() or not self.settings.enable_cache:
            return False
        
        try:
            ttl = ttl or self.settings.cache_ttl_seconds
            serialized_value = json.dumps(value, default=str)
            self.client.setex(key, timedelta(seconds=ttl), serialized_value)
            return True
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self.is_connected() or not self.settings.enable_cache:
            return None
        
        try:
            cached_value = self.client.get(key)
            if cached_value:
                return json.loads(cached_value)
            return None
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """Delete a cache entry."""
        if not self.is_connected():
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False
    
    def cache_search_results(self, query: str, results: List[Dict], ttl: Optional[int] = None) -> bool:
        """Cache search results for a query."""
        cache_key = self._generate_cache_key("search", query)
        return self.set_cache(cache_key, results, ttl)
    
    def get_cached_search_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached search results for a query."""
        cache_key = self._generate_cache_key("search", query)
        return self.get_cache(cache_key)
    
    def cache_embedding(self, text: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Cache embedding for a text."""
        cache_key = self._generate_cache_key("embedding", text)
        return self.set_cache(cache_key, embedding, ttl)
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for a text."""
        cache_key = self._generate_cache_key("embedding", text)
        return self.get_cache(cache_key)
    
    def cache_document_metadata(self, doc_id: str, metadata: Dict, ttl: Optional[int] = None) -> bool:
        """Cache document metadata."""
        cache_key = self._generate_cache_key("doc_meta", doc_id)
        return self.set_cache(cache_key, metadata, ttl)
    
    def get_cached_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get cached document metadata."""
        cache_key = self._generate_cache_key("doc_meta", doc_id)
        return self.get_cache(cache_key)
    
    def cache_rag_response(self, query: str, context: str, response: str, ttl: Optional[int] = None) -> bool:
        """Cache RAG response."""
        cache_key = self._generate_cache_key("rag_response", query, context[:100])  # Limit context for key
        response_data = {
            "query": query,
            "context": context,
            "response": response,
            "timestamp": str(datetime.now())
        }
        return self.set_cache(cache_key, response_data, ttl)
    
    def get_cached_rag_response(self, query: str, context: str) -> Optional[Dict]:
        """Get cached RAG response."""
        cache_key = self._generate_cache_key("rag_response", query, context[:100])
        return self.get_cache(cache_key)
    
    def clear_all_cache(self) -> bool:
        """Clear all cache entries."""
        if not self.is_connected():
            return False
        
        try:
            self.client.flushdb()
            logger.info("Cleared all cache entries")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            info = self.client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"connected": False, "error": str(e)}