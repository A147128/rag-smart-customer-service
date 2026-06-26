"""
Cache Manager --- Redis Cache-Aside with defense-in-depth
=========================================================
Design pattern: Cache-Aside
Defenses: penetration, breakdown, avalanche, Big Key, graceful degradation

Usage:
    import redis.asyncio as aioredis
    from cache_manager import CacheConfig, CacheManager

    cfg = CacheConfig()
    redis = aioredis.from_url(cfg.redis_url, decode_responses=True)
    cache = CacheManager(redis, cfg)
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import redis.asyncio as aioredis
import redis.exceptions
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ================================================================
# 1. Configuration
# ================================================================


class CacheConfig(BaseSettings):
    """Cache configuration with sensible defaults. Supports .env override."""

    model_config = SettingsConfigDict(
        env_prefix="CACHE_",
        case_sensitive=False,
    )

    # Connection
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # TTL (seconds)
    default_ttl: int = Field(default=3600, ge=1, description="Default cache TTL (seconds)")
    null_ttl: int = Field(default=60, ge=10, description="TTL for null-marker (anti-penetration)")
    lock_ttl: int = Field(default=5, ge=1, description="Distributed lock TTL (anti-breakdown)")

    # Anti-avalanche jitter (%)
    ttl_jitter_pct: int = Field(default=20, ge=0, le=50, description="TTL random jitter +/- percent")

    # Anti-breakdown retry
    lock_retry_delay: float = Field(default=0.1, description="Retry delay when lock fails")
    lock_max_retries: int = Field(default=30, description="Max retries for lock")

    # Big Key
    big_key_threshold: int = Field(default=10240, description="Size threshold (bytes) for Big Key warning")

    # Key naming
    key_prefix: str = Field(
        default="rag:response:",
        description="Redis key prefix (project:module:)",
    )
    lock_key_prefix: str = Field(
        default="rag:lock:",
        description="Redis lock key prefix",
    )
    null_marker: str = Field(
        default="__NULL__",
        description="Value to mark non-existent data (anti-penetration)",
    )


# ================================================================
# 2. Cache Manager
# ================================================================


class CacheManager:
    """Redis Cache-Aside manager with defense-in-depth.

    All public methods gracefully degrade on Redis failure.
    """

    __slots__ = ("_redis", "_cfg")

    def __init__(self, redis: aioredis.Redis, config: CacheConfig) -> None:
        self._redis = redis
        self._cfg = config

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    async def get_or_set(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        ttl: int | None = None,
        serializer: Callable[[Any], str] | None = None,
        deserializer: Callable[[str], Any] | None = None,
    ) -> Any:
        """Cache-Aside read with anti-penetration, anti-breakdown, anti-avalanche.

        Penetration:  caches __NULL__ if loader returns None/empty.
        Breakdown:    uses SETNX lock so only one worker rebuilds.
        Avalanche:    TTL is randomly jittered.
        Degradation:  on RedisError falls through to loader directly.
        """
        full_key = self._full_key(key)
        base_ttl = ttl or self._cfg.default_ttl

        # 1. Read cache
        cached = await self._safe_redis_get(full_key)
        if cached is not None:
            return self._unwrap_null(cached, deserializer)

        # 2. Cache miss --- try to acquire rebuild lock
        lock_key = f"{self._cfg.lock_key_prefix}{key}"
        locked = await self._try_acquire_lock(lock_key)
        if locked:
            return await self._rebuild_and_cache(key, full_key, loader, base_ttl, serializer)

        # 3. Lock not acquired --- another worker is rebuilding; retry cache read
        for _ in range(self._cfg.lock_max_retries):
            await asyncio.sleep(self._cfg.lock_retry_delay)
            cached = await self._safe_redis_get(full_key)
            if cached is not None:
                return self._unwrap_null(cached, deserializer)

        # 4. Exhausted retries --- fall through to loader as last resort
        logger.warning("Lock wait exhausted for key=%s, falling through to loader", key)
        return await self._load_and_return(loader, key)

    async def invalidate(self, key: str) -> None:
        """Delete a single cache entry (write-through invalidation)."""
        await self._safe_redis_delete(self._full_key(key))

    async def safe_delete(self, pattern: str, batch_size: int = 100) -> int:
        """Batch delete keys matching a glob pattern using SCAN (non-blocking)."""
        total = 0
        cursor = 0
        while True:
            cursor, keys = await self._safe_redis_scan(cursor, pattern, batch_size)
            if keys:
                await self._safe_redis_delete(*keys)
                total += len(keys)
            if cursor == 0:
                break
        return total

    # ------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        """Build the full Redis key with prefix."""
        return f"{self._cfg.key_prefix}{key}"

    @staticmethod
    def _jitter_ttl(base_ttl: int, pct: int) -> int:
        """Add random jitter to TTL to prevent avalanche."""
        if pct <= 0:
            return base_ttl
        delta = int(base_ttl * pct / 100)
        return max(1, base_ttl + random.randint(-delta, delta))

    def _unwrap_null(self, raw: str, deserializer: Callable[[str], Any] | None) -> Any:
        if raw == self._cfg.null_marker:
            return None
        return deserializer(raw) if deserializer else raw

    async def _rebuild_and_cache(
        self,
        key: str,
        full_key: str,
        loader: Callable[[], Awaitable[Any]],
        base_ttl: int,
        serializer: Callable[[Any], str] | None,
    ) -> Any:
        """Load data, serialize, write cache with jittered TTL."""
        value = await self._load_and_return(loader, key)
        ttl = self._jitter_ttl(base_ttl, self._cfg.ttl_jitter_pct)

        # Anti-penetration: empty result gets a shorter TTL
        if value in (None, [], {}):
            await self._safe_redis_set(full_key, self._cfg.null_marker, self._cfg.null_ttl)
        else:
            raw = serializer(value) if serializer else str(value)
            self._warn_big_key(key, raw)
            await self._safe_redis_set(full_key, raw, ttl)
        return value

    async def _load_and_return(self, loader: Callable[[], Awaitable[Any]], key: str | None) -> Any:
        try:
            return await loader()
        except Exception:
            logger.exception("Loader failed for key=%s", key)
            return None

    async def _try_acquire_lock(self, lock_key: str) -> bool:
        try:
            acquired = await self._redis.set(lock_key, "1", nx=True, ex=self._cfg.lock_ttl)
            return bool(acquired)
        except redis.exceptions.RedisError:
            logger.exception("Lock acquisition failed for %s", lock_key)
            return False

    # ------------------------------------------------------------
    # Safe Redis operations (degradation boundary)
    # ------------------------------------------------------------

    async def _safe_redis_get(self, key: str) -> str | None:
        try:
            return await self._redis.get(key)
        except redis.exceptions.RedisError:
            logger.warning("Redis GET failed for %s, degraded", key, exc_info=True)
            return None

    async def _safe_redis_set(self, key: str, value: str, ttl: int) -> None:
        try:
            await self._redis.setex(key, ttl, value)
        except redis.exceptions.RedisError:
            logger.warning("Redis SET failed for %s, degraded", key, exc_info=True)

    async def _safe_redis_delete(self, *keys: str) -> None:
        if keys:
            try:
                await self._redis.delete(*keys)
            except redis.exceptions.RedisError:
                logger.warning("Redis DELETE failed for %d keys, degraded", len(keys), exc_info=True)

    async def _safe_redis_scan(self, cursor: int, pattern: str, count: int) -> tuple[int, list[str]]:
        try:
            result = await self._redis.scan(cursor=cursor, match=pattern, count=count)
            if not result:
                return 0, []
            return result[0], result[1]
        except redis.exceptions.RedisError:
            logger.warning("Redis SCAN failed for %s, degraded", pattern, exc_info=True)
            return 0, []

    def _warn_big_key(self, logical_key: str, raw: str) -> None:
        size = len(raw.encode("utf-8"))
        if size >= self._cfg.big_key_threshold:
            logger.warning(
                "Big Key detected: key=%s, size=%d bytes, threshold=%d bytes",
                logical_key,
                size,
                self._cfg.big_key_threshold,
            )


# ================================================================
# 3. Convenience factory
# ================================================================


async def create_cache_manager(
    redis_url: str | None = None,
    key_prefix: str = "rag:response:",
) -> CacheManager:
    """Create a CacheManager with connected Redis."""
    cfg = CacheConfig(
        redis_url=redis_url or "redis://localhost:6379/0",
        key_prefix=key_prefix,
    )
    redis = aioredis.from_url(cfg.redis_url, decode_responses=True)
    return CacheManager(redis, cfg)
