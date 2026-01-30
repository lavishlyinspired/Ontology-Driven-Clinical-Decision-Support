"""
Caching Layer for Performance Optimization

Implements multi-level caching:
- Memory cache for hot data
- Redis for distributed caching
- Result caching for expensive operations
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import asyncio
from collections import OrderedDict

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)


class LRUCache:
    """In-memory LRU cache for frequently accessed data."""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to cache
        """
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        
        # Check expiration
        value, expiry = self.cache[key]
        if expiry and datetime.now() > expiry:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache with optional TTL."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        expiry = None
        if ttl_seconds:
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        
        self.cache[key] = (value, expiry)
        self.cache.move_to_end(key)
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }


class ResultCache:
    """Cache for expensive operation results (LCA analysis, ontology queries)."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize result cache.
        
        Args:
            default_ttl: Default TTL in seconds (1 hour)
        """
        # Separate caches for different data types
        self.patient_cache = LRUCache(capacity=500)
        self.ontology_cache = LRUCache(capacity=2000)
        self.analysis_cache = LRUCache(capacity=500)
        self.guideline_cache = LRUCache(capacity=100)
        
        self.default_ttl = default_ttl
    
    def cache_patient_data(
        self, 
        patient_id: str, 
        patient_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache patient demographic and clinical data."""
        self.patient_cache.set(
            f"patient:{patient_id}",
            patient_data,
            ttl or self.default_ttl
        )
    
    def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached patient data."""
        return self.patient_cache.get(f"patient:{patient_id}")
    
    def cache_analysis_result(
        self,
        patient_id: str,
        complexity: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache LCA analysis result."""
        cache_key = f"analysis:{patient_id}:{complexity}"
        self.analysis_cache.set(cache_key, result, ttl or self.default_ttl)
    
    def get_analysis_result(
        self,
        patient_id: str,
        complexity: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result."""
        cache_key = f"analysis:{patient_id}:{complexity}"
        return self.analysis_cache.get(cache_key)
    
    def cache_ontology_query(
        self,
        query_hash: str,
        results: Any,
        ttl: Optional[int] = None
    ):
        """Cache ontology SPARQL query results."""
        self.ontology_cache.set(
            f"ontology:{query_hash}",
            results,
            ttl or (self.default_ttl * 24)  # Ontology queries cached longer
        )
    
    def get_ontology_query(self, query_hash: str) -> Optional[Any]:
        """Retrieve cached ontology query results."""
        return self.ontology_cache.get(f"ontology:{query_hash}")
    
    def cache_guideline(
        self,
        guideline_id: str,
        guideline_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache guideline rules."""
        self.guideline_cache.set(
            f"guideline:{guideline_id}",
            guideline_data,
            ttl or (self.default_ttl * 168)  # Guidelines cached 1 week
        )
    
    def get_guideline(self, guideline_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached guideline."""
        return self.guideline_cache.get(f"guideline:{guideline_id}")
    
    def invalidate_patient(self, patient_id: str):
        """Invalidate all cache entries for a patient."""
        self.patient_cache.delete(f"patient:{patient_id}")
        
        # Invalidate all analysis results for this patient
        for complexity in ['SIMPLE', 'MODERATE', 'COMPLEX', 'CRITICAL']:
            self.analysis_cache.delete(f"analysis:{patient_id}:{complexity}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'patient_cache': self.patient_cache.stats(),
            'ontology_cache': self.ontology_cache.stats(),
            'analysis_cache': self.analysis_cache.stats(),
            'guideline_cache': self.guideline_cache.stats()
        }


def cache_result(
    cache_instance: ResultCache,
    cache_type: str = 'analysis',
    key_generator: Optional[Callable] = None,
    ttl: Optional[int] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_instance: ResultCache instance
        cache_type: Type of cache ('analysis', 'patient', 'ontology', 'guideline')
        key_generator: Function to generate cache key from args
        ttl: Time to live in seconds
        
    Example:
        @cache_result(result_cache, cache_type='analysis')
        async def analyze_patient(patient_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default: hash all args
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            if cache_type == 'analysis':
                cached = cache_instance.analysis_cache.get(cache_key)
            elif cache_type == 'patient':
                cached = cache_instance.patient_cache.get(cache_key)
            elif cache_type == 'ontology':
                cached = cache_instance.ontology_cache.get(cache_key)
            elif cache_type == 'guideline':
                cached = cache_instance.guideline_cache.get(cache_key)
            else:
                cached = None
            
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_type == 'analysis':
                cache_instance.analysis_cache.set(cache_key, result, ttl)
            elif cache_type == 'patient':
                cache_instance.patient_cache.set(cache_key, result, ttl)
            elif cache_type == 'ontology':
                cache_instance.ontology_cache.set(cache_key, result, ttl)
            elif cache_type == 'guideline':
                cache_instance.guideline_cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Same logic for sync functions
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            if cache_type == 'analysis':
                cached = cache_instance.analysis_cache.get(cache_key)
            elif cache_type == 'patient':
                cached = cache_instance.patient_cache.get(cache_key)
            elif cache_type == 'ontology':
                cached = cache_instance.ontology_cache.get(cache_key)
            elif cache_type == 'guideline':
                cached = cache_instance.guideline_cache.get(cache_key)
            else:
                cached = None
            
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            
            if cache_type == 'analysis':
                cache_instance.analysis_cache.set(cache_key, result, ttl)
            elif cache_type == 'patient':
                cache_instance.patient_cache.set(cache_key, result, ttl)
            elif cache_type == 'ontology':
                cache_instance.ontology_cache.set(cache_key, result, ttl)
            elif cache_type == 'guideline':
                cache_instance.guideline_cache.set(cache_key, result, ttl)
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class SessionCache:
    """Cache for chat sessions and conversation state."""
    
    def __init__(self, default_ttl: int = 1800):
        """
        Initialize session cache.
        
        Args:
            default_ttl: Default session TTL (30 minutes)
        """
        self.cache = LRUCache(capacity=1000)
        self.default_ttl = default_ttl
    
    def save_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Save session data."""
        self.cache.set(session_id, session_data, ttl or self.default_ttl)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        return self.cache.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete session."""
        self.cache.delete(session_id)
    
    def extend_session(self, session_id: str, ttl: Optional[int] = None):
        """Extend session TTL."""
        session_data = self.cache.get(session_id)
        if session_data:
            self.cache.set(session_id, session_data, ttl or self.default_ttl)


# Global cache instances
result_cache = ResultCache()
session_cache = SessionCache()


# Utility functions

def generate_patient_cache_key(patient_id: str) -> str:
    """Generate cache key for patient."""
    return f"patient:{patient_id}"


def generate_analysis_cache_key(patient_data: Dict[str, Any]) -> str:
    """
    Generate cache key for analysis based on patient data hash.
    
    Uses demographic, clinical, and biomarker data.
    """
    key_data = {
        'demographics': patient_data.get('demographics', {}),
        'diagnosis': patient_data.get('diagnosis', {}),
        'biomarkers': patient_data.get('biomarkers', {}),
        'comorbidities': sorted(patient_data.get('comorbidities', [])),
        'performance_status': patient_data.get('performance_status')
    }
    
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()


def generate_ontology_query_hash(sparql_query: str) -> str:
    """Generate hash for SPARQL query."""
    return hashlib.md5(sparql_query.encode()).hexdigest()
