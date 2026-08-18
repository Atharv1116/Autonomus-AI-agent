"""
Query result caching.

Provides an in-memory LRU cache with TTL expiration for repeated
queries. Cache key is derived from question text + schema hash
to ensure invalidation when the database schema changes.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional

from config.logging_config import get_logger

logger = get_logger("utils.cache")


class QueryCache:
    """
    LRU cache with TTL expiration for query results.

    Thread-safe for single-threaded async use (Streamlit).
    Keys are computed from question + schema hash to auto-invalidate
    when the database schema changes.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600) -> None:
        """
        Initialize the query cache.

        Args:
            max_size: Maximum number of cached entries.
            ttl_seconds: Time-to-live for each entry in seconds.
        """
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        logger.info("QueryCache initialized (max_size=%d, ttl=%ds)", max_size, ttl_seconds)

    @staticmethod
    def _compute_key(question: str, schema_hash: str = "") -> str:
        """
        Compute a cache key from question and schema hash.

        Args:
            question: The user's natural language question.
            schema_hash: Hash of the current database schema.

        Returns:
            SHA256 hash string used as cache key.
        """
        content = f"{question.strip().lower()}:{schema_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, question: str, schema_hash: str = "") -> Optional[dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            question: The user's question.
            schema_hash: Current schema hash.

        Returns:
            Cached result dictionary, or None if not found / expired.
        """
        key = self._compute_key(question, schema_hash)

        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check TTL expiration
        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            self._misses += 1
            logger.debug("Cache entry expired: %s", key[:12])
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        logger.debug("Cache hit: %s", key[:12])
        return entry["data"]

    def set(self, question: str, data: dict[str, Any], schema_hash: str = "") -> None:
        """
        Store a result in the cache.

        Args:
            question: The user's question.
            data: The result data to cache.
            schema_hash: Current schema hash.
        """
        key = self._compute_key(question, schema_hash)

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Cache eviction: %s", evicted_key[:12])

        self._cache[key] = {
            "data": data,
            "timestamp": time.time(),
            "question": question,
        }
        logger.debug("Cache set: %s", key[:12])

    def invalidate(self, question: Optional[str] = None, schema_hash: str = "") -> None:
        """
        Invalidate cache entries.

        Args:
            question: If provided, invalidate only this question's entry.
                      If None, clear the entire cache.
            schema_hash: Schema hash for targeted invalidation.
        """
        if question is None:
            self._cache.clear()
            logger.info("Cache fully invalidated")
        else:
            key = self._compute_key(question, schema_hash)
            if key in self._cache:
                del self._cache[key]
                logger.debug("Cache entry invalidated: %s", key[:12])

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 1),
            "ttl_seconds": self._ttl,
        }

    def __len__(self) -> int:
        return len(self._cache)
