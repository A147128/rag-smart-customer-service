"""Functional tests for cache_manager.py

Covers all 6 defense mechanisms:
1. Cache penetration (null marker)
2. Cache breakdown (SETNX lock)
3. Cache avalanche (TTL jitter)
4. Big Key warning
5. Graceful degradation on RedisError
6. Safe batch delete with SCAN
"""

from __future__ import annotations

import logging
from typing import NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cache.cache_manager import CacheConfig, CacheManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Config with deterministic TTL (no jitter) for testing."""
    return CacheConfig(
        default_ttl=10,
        null_ttl=10,
        lock_ttl=3,
        ttl_jitter_pct=0,
        key_prefix="test:response:",
        lock_key_prefix="test:lock:",
        null_marker="__NULL__",
    )


@pytest.fixture
def mock_redis():
    """Mock redis.asyncio.Redis with async methods."""
    redis = MagicMock()
    redis.get = AsyncMock()
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.set = AsyncMock()
    redis.scan = AsyncMock()
    return redis


@pytest.fixture
def cache(mock_redis, cfg):
    return CacheManager(mock_redis, cfg)


# ---------------------------------------------------------------------------
# 1. Anti-Penetration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_marker_cached_when_loader_returns_none(cache, mock_redis, cfg) -> None:
    """When loader returns None, NULL marker is cached with null_ttl."""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True  # lock acquired

    async def loader() -> None:
        return None

    result = await cache.get_or_set("key1", loader)

    assert result is None
    # Verify null marker was written
    setex_calls = [c for c in mock_redis.setex.call_args_list]
    assert any(cfg.null_marker in str(c) for c in setex_calls)


@pytest.mark.asyncio
async def test_null_marker_cached_when_loader_returns_empty(cache, mock_redis) -> None:
    """When loader returns empty list, NULL marker is cached."""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    async def loader():
        return []

    result = await cache.get_or_set("key2", loader)
    assert result == []

    setex_calls = mock_redis.setex.call_args_list
    assert len(setex_calls) >= 1


@pytest.mark.asyncio
async def test_null_marker_returned_as_none(cache, mock_redis, cfg) -> None:
    """When cache returns NULL marker, get_or_set returns None."""
    mock_redis.get.return_value = cfg.null_marker

    async def loader() -> str:
        return "should_not_call"

    result = await cache.get_or_set("key3", loader)
    assert result is None


# ---------------------------------------------------------------------------
# 2. Anti-Breakdown (SETNX lock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_lock(cache, mock_redis) -> None:
    """Cache hit returns immediately without acquiring lock."""
    mock_redis.get.return_value = "cached_value"

    async def loader() -> NoReturn:
        raise RuntimeError("should not be called")

    result = await cache.get_or_set("key4", loader)
    assert result == "cached_value"
    # set() (lock) should NOT have been called
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss_acquires_lock_and_rebuilds(cache, mock_redis) -> None:
    """On miss, lock acquired → loader called → result cached."""
    mock_redis.get.return_value = None  # initial miss
    mock_redis.set.return_value = True  # lock acquired

    async def loader() -> str:
        return "fresh_data"

    result = await cache.get_or_set("key5", loader)
    assert result == "fresh_data"
    mock_redis.set.assert_called_once()  # lock
    mock_redis.setex.assert_called_once()  # cache write


@pytest.mark.asyncio
async def test_lock_not_acquired_retries_cache_read(cache, mock_redis) -> None:
    """If lock is held by another worker, retry reading cache."""
    # First get returns None (miss), lock fails, then cache populated
    mock_redis.get.side_effect = [None, None, "other_worker_data"]
    mock_redis.set.return_value = False  # lock not acquired

    async def loader() -> str:
        return "should_not_call"

    # Use a shorter max_retries for test speed
    cache._cfg = CacheConfig(
        default_ttl=10,
        null_ttl=10,
        lock_ttl=3,
        ttl_jitter_pct=0,
        lock_max_retries=3,
        lock_retry_delay=0.01,
    )
    result = await cache.get_or_set("key6", loader)
    assert result == "other_worker_data"
    mock_redis.set.assert_called_once()  # tried lock once


# ---------------------------------------------------------------------------
# 3. Anti-Avalanche (TTL jitter)
# ---------------------------------------------------------------------------


def test_jitter_ttl_in_range() -> None:
    """Jitter stays within +/- pct of base TTL."""
    base = 100
    pct = 20
    # Run many times to verify range
    results = [CacheManager._jitter_ttl(base, pct) for _ in range(200)]
    assert all(80 <= r <= 120 for r in results)
    # Should have some variation across 200 samples
    assert len(set(results)) > 1


def test_jitter_ttl_zero_pct() -> None:
    """Zero jitter returns exact base TTL."""
    assert CacheManager._jitter_ttl(100, 0) == 100


def test_jitter_ttl_minimum_one() -> None:
    """Jitter never goes below 1."""
    assert CacheManager._jitter_ttl(1, 50) == 1


# ---------------------------------------------------------------------------
# 4. Big Key detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_big_key_warning_logged(cache, mock_redis, caplog, cfg) -> None:
    """When serialized value exceeds threshold, WARNING is logged."""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    big_data = "x" * (cfg.big_key_threshold + 10)

    async def loader():
        return big_data

    with caplog.at_level(logging.WARNING):
        await cache.get_or_set("key7", loader, serializer=lambda x: x)

    assert any("Big Key" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 5. Graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_error_falls_through_to_loader(cache, mock_redis) -> None:
    """When Redis fails, get_or_set falls through to loader directly."""
    import redis.exceptions

    mock_redis.get.side_effect = redis.exceptions.ConnectionError("down")
    mock_redis.set.return_value = True

    async def loader() -> str:
        return "fallback_data"

    result = await cache.get_or_set("key8", loader)
    assert result == "fallback_data"


@pytest.mark.asyncio
async def test_redis_error_on_set_returns_loader_value(cache, mock_redis) -> None:
    """When Redis set fails after loader, still returns loader value."""
    import redis.exceptions

    mock_redis.get.side_effect = [None, None, None]  # miss + retry
    mock_redis.set.return_value = True
    mock_redis.setex.side_effect = redis.exceptions.ConnectionError("down")

    async def loader() -> str:
        return "still_returned"

    cache._cfg = CacheConfig(
        default_ttl=10,
        null_ttl=10,
        lock_ttl=3,
        ttl_jitter_pct=0,
        lock_max_retries=2,
        lock_retry_delay=0.01,
    )
    result = await cache.get_or_set("key9", loader)
    assert result == "still_returned"


# ---------------------------------------------------------------------------
# 6. Invalidate & Safe Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_deletes_key(cache, mock_redis) -> None:
    """invalidate calls Redis DELETE with the correct key."""
    await cache.invalidate("mykey")
    mock_redis.delete.assert_called_once_with("test:response:mykey")


@pytest.mark.asyncio
async def test_safe_delete_scans_and_deletes(cache, mock_redis) -> None:
    """safe_delete uses SCAN to find and delete matching keys."""
    mock_redis.scan.side_effect = [
        (0, ["test:response:a", "test:response:b"]),  # first batch, cursor=0 done
    ]

    count = await cache.safe_delete("foo*")
    assert count == 2
    mock_redis.delete.assert_called_once_with("test:response:a", "test:response:b")


@pytest.mark.asyncio
async def test_safe_delete_multiple_batches(cache, mock_redis) -> None:
    """safe_delete handles multiple SCAN batches."""
    mock_redis.scan.side_effect = [
        (1, ["test:response:a"]),
        (0, ["test:response:b"]),
    ]
    count = await cache.safe_delete("bar*")
    assert count == 2
    assert mock_redis.delete.call_count == 2


@pytest.mark.asyncio
async def test_safe_delete_empty(cache, mock_redis) -> None:
    """safe_delete with no matches returns 0."""
    mock_redis.scan.return_value = (0, [])
    count = await cache.safe_delete("no_match")
    assert count == 0


# ---------------------------------------------------------------------------
# 7. Serializer / Deserializer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serializer_deserializer_roundtrip(cache, mock_redis) -> None:
    """Custom serializer/deserializer round-trips correctly."""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    async def loader():
        return {"answer": 42}

    result = await cache.get_or_set(
        "key10",
        loader,
        serializer=lambda x: str(x["answer"]),
        deserializer=lambda s: {"answer": int(s)},
    )
    assert result == {"answer": 42}


# ---------------------------------------------------------------------------
# 8. Factory function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_cache_manager_default_prefix() -> None:
    """Factory creates CacheManager with default prefix."""
    from cache.cache_manager import create_cache_manager

    with patch("cache.cache_manager.aioredis.from_url") as mock_from_url:
        mock_from_url.return_value = MagicMock()
        mgr = await create_cache_manager()
        assert isinstance(mgr, CacheManager)
        assert mgr._cfg.key_prefix == "rag:response:"
