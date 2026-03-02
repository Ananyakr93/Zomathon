"""
Cache Manager — CSAO Production
=================================
Multi-tier caching with L1 (in-process LRU) and L2 (Redis-compatible
dict store for local testing).  Tracks hit rates for monitoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  L1 — IN-PROCESS LRU CACHE (per pod)
# ═════════════════════════════════════════════════════════════════════════════

class L1Cache:
    """Thread-safe-ish LRU cache with TTL expiration."""

    def __init__(self, max_items: int = 10_000, default_ttl_s: int = 60):
        self._max_items = max_items
        self._default_ttl = default_ttl_s
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._store[key]
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        expiry = time.monotonic() + ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expiry)
        while len(self._store) > self._max_items:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_items": self._max_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  L2 — REDIS WITH AUTO-FALLBACK TO IN-MEMORY DICT
# ═════════════════════════════════════════════════════════════════════════════

class L2Cache:
    """
    Redis-backed L2 cache with automatic fallback to in-memory dict.

    On init, attempts to connect to Redis. If Redis is unavailable,
    silently falls back to a Python dict store (zero-config local dev).

    Env vars:
        REDIS_URL : Redis connection URL (default: redis://localhost:6379/0)
    """

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._using_redis = False
        self._redis = None
        self._store: dict[str, tuple[Any, float]] = {}

        # Auto-detect Redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis.from_url(redis_url, socket_timeout=1)
            self._redis.ping()
            self._using_redis = True
            logger.info("L2Cache: Connected to Redis at %s", redis_url)
        except Exception:
            self._redis = None
            self._using_redis = False
            logger.info("L2Cache: Redis unavailable, using in-memory dict fallback")

    @property
    def backend(self) -> str:
        return "redis" if self._using_redis else "in_memory"

    def get(self, key: str) -> Any | None:
        if self._using_redis:
            try:
                raw = self._redis.get(key)  # type: ignore
                if raw is None:
                    self._misses += 1
                    return None
                self._hits += 1
                return json.loads(raw)
            except Exception:
                self._misses += 1
                return None
        else:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl_s: int = 300) -> None:
        if self._using_redis:
            try:
                self._redis.setex(key, ttl_s, json.dumps(value, default=str))  # type: ignore
            except Exception:
                logger.debug("Redis set failed for key %s, using in-memory", key)
                expiry = time.monotonic() + ttl_s
                self._store[key] = (value, expiry)
        else:
            expiry = time.monotonic() + ttl_s
            self._store[key] = (value, expiry)

    def delete(self, key: str) -> None:
        if self._using_redis:
            try:
                self._redis.delete(key)  # type: ignore
            except Exception:
                pass
        self._store.pop(key, None)

    def flush(self) -> None:
        if self._using_redis:
            try:
                self._redis.flushdb()  # type: ignore
            except Exception:
                pass
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "size": len(self._store) if not self._using_redis else "N/A (Redis)",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }



# ═════════════════════════════════════════════════════════════════════════════
#  CACHE MANAGER — UNIFIED INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheTTLs:
    """Centralised TTL configuration (seconds)."""
    menu: int = 3600           # 1 hour
    user_embedding: int = 300   # 5 min
    llm_response: int = 1800    # 30 min
    restaurant_meta: int = 7200 # 2 hours
    candidate_set: int = 120    # 2 min
    feature_vector: int = 60    # 1 min


class CacheManager:
    """
    Two-tier cache with namespace-based key routing.

    L1 (in-process) for ultra-hot data (< 1ms lookup).
    L2 (Redis-like)  for shared data across pods.

    Usage:
        cm = CacheManager()
        cm.get_or_set("menu:R123", fetch_fn=lambda: db.get_menu("R123"),
                      namespace="menu")
    """

    def __init__(
        self,
        l1_max: int = 10_000,
        l1_ttl: int = 60,
        ttls: CacheTTLs | None = None,
    ) -> None:
        self.l1 = L1Cache(max_items=l1_max, default_ttl_s=l1_ttl)
        self.l2 = L2Cache()
        self.ttls = ttls or CacheTTLs()

    def get_or_set(
        self,
        key: str,
        fetch_fn: Any = None,
        namespace: str = "default",
    ) -> Any | None:
        """
        Two-tier lookup:
          1. Check L1
          2. If miss, check L2
          3. If miss, call fetch_fn and populate both tiers
        """
        # L1
        val = self.l1.get(key)
        if val is not None:
            return val

        # L2
        val = self.l2.get(key)
        if val is not None:
            self.l1.put(key, val, ttl_s=self._l1_ttl_for(namespace))
            return val

        # Fetch
        if fetch_fn is None:
            return None
        val = fetch_fn()
        if val is not None:
            l2_ttl = self._l2_ttl_for(namespace)
            self.l2.set(key, val, ttl_s=l2_ttl)
            self.l1.put(key, val, ttl_s=self._l1_ttl_for(namespace))
        return val

    def invalidate(self, key: str) -> None:
        self.l1.invalidate(key)
        self.l2.delete(key)

    def _l2_ttl_for(self, namespace: str) -> int:
        return getattr(self.ttls, namespace, 300)

    def _l1_ttl_for(self, namespace: str) -> int:
        # L1 TTL is always shorter than L2
        return min(60, self._l2_ttl_for(namespace) // 4)

    def stats(self) -> dict:
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
        }

    @staticmethod
    def make_cache_key(namespace: str, **kwargs) -> str:
        """Generate a deterministic cache key from namespace + params."""
        raw = json.dumps(kwargs, sort_keys=True, default=str)
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{namespace}:{h}"
