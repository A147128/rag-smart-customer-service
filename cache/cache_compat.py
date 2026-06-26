"""
同步版 Redis 缓存 - 兼容 ResponseCache 接口
直接使用 redis.Redis（同步客户端），不依赖异步桥接

用法（替换 cache_store.ResponseCache）：
    from cache_compat import ResponseCacheRedis

    self.cache = ResponseCacheRedis(redis_url="redis://192.168.247.129:6379/0")
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from collections.abc import Callable

import redis
import redis.exceptions

logger = logging.getLogger(__name__)


class ResponseCacheRedis:
    """同步版 Redis 缓存，兼容 ResponseCache 接口

    防御特性：
    - 缓存穿透：空值标记 + 短 TTL
    - 缓存击穿：SETNX 分布式锁
    - 缓存雪崩：TTL 随机抖动 ±20%
    - Big Key 预警：超过 10KB 记录 WARN
    - 异常降级：Redis 异常时静默降级
    """

    def __init__(
        self,
        cache_file: str = "./response_cache.json",  # 兼容旧接口，实际不用
        ttl_hours: int = 24,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "rag:response:",
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._lock_prefix = "rag:lock:"
        self._default_ttl = ttl_hours * 3600
        self._null_ttl = 60  # 空值缓存 60 秒
        self._lock_ttl = 5  # 锁超时 5 秒
        self._lock_retry_delay = 0.1
        self._lock_max_retries = 30
        self._big_key_threshold = 10240  # 10KB
        self._null_marker = "__NULL__"
        self._ttl_jitter_pct = 20  # ±20% 抖动

        self._redis: redis.Redis | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    # 连接管理
    # ----------------------------------------------------------------

    def _ensure_redis(self) -> redis.Redis:
        """惰性初始化 Redis 连接"""
        if self._redis is not None:
            return self._redis
        with self._lock:
            if self._redis is not None:
                return self._redis
            try:
                self._redis = redis.from_url(self._redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Redis 连接成功: %s", self._redis_url)
            except redis.exceptions.RedisError as e:
                logger.error("Redis 连接失败: %s, 降级运行（无缓存）", e)
                self._redis = None
            return self._redis

    # ----------------------------------------------------------------
    # 公共 API（与 ResponseCache 完全兼容）
    # ----------------------------------------------------------------

    def _generate_key(self, question: str) -> str:
        """生成 MD5 哈希键"""
        return hashlib.md5(question.encode("utf-8")).hexdigest()

    def get(self, question: str) -> str | None:
        """获取缓存响应"""
        key = self._full_key(self._generate_key(question))
        r = self._ensure_redis()
        if r is None:
            return None  # Redis 不可用，降级

        try:
            cached = r.get(key)
            if cached is None:
                return None
            # 空值标记
            if cached == self._null_marker:
                return None
            return cached
        except redis.exceptions.RedisError:
            logger.warning("Redis GET 失败, 降级", exc_info=True)
            return None

    def set(self, question: str, response: str) -> None:
        """设置缓存（带雪崩防护：随机 TTL）"""
        raw_key = self._generate_key(question)
        key = self._full_key(raw_key)
        r = self._ensure_redis()
        if r is None:
            return

        try:
            # Big Key 预警
            size = len(response.encode("utf-8"))
            if size >= self._big_key_threshold:
                logger.warning("Big Key 检测: key=%s, size=%d bytes", raw_key, size)

            # 雪崩防护：随机 TTL 抖动
            ttl = self._jitter_ttl(self._default_ttl)
            r.setex(key, ttl, response)
            logger.debug("缓存写入: key=%s, ttl=%ds", raw_key, ttl)
        except redis.exceptions.RedisError:
            logger.warning("Redis SET 失败, 降级", exc_info=True)

    def get_or_set(
        self,
        question: str,
        loader: Callable[[], str | None],
    ) -> str | None:
        """缓存穿透/击穿防护的 get_or_set（旁路缓存模式）

        流程：查缓存 → 命中返回 → 尝试锁 → 重建缓存 → 返回
        """
        raw_key = self._generate_key(question)
        key = self._full_key(raw_key)
        r = self._ensure_redis()
        if r is None:
            return loader()

        try:
            # 1. 查缓存
            cached = r.get(key)
            if cached is not None:
                if cached == self._null_marker:
                    return None
                return cached

            # 2. 缓存未命中，尝试获取分布式锁（防击穿）
            lock_key = f"{self._lock_prefix}{raw_key}"
            locked = r.set(lock_key, "1", nx=True, ex=self._lock_ttl)
            if locked:
                try:
                    # 3. 重建缓存
                    value = loader()
                    if value is None:
                        # 穿透防护：空值标记，短 TTL
                        r.setex(key, self._null_ttl, self._null_marker)
                    else:
                        # Big Key 预警
                        size = len(value.encode("utf-8"))
                        if size >= self._big_key_threshold:
                            logger.warning("Big Key 检测: key=%s, size=%d bytes", raw_key, size)
                        # 雪崩防护：随机 TTL
                        ttl = self._jitter_ttl(self._default_ttl)
                        r.setex(key, ttl, value)
                    return value
                finally:
                    try:
                        r.delete(lock_key)
                    except redis.exceptions.RedisError:
                        pass
            else:
                # 4. 锁被其他进程持有，等待后重试
                for _ in range(self._lock_max_retries):
                    import time

                    time.sleep(self._lock_retry_delay)
                    cached = r.get(key)
                    if cached is not None:
                        if cached == self._null_marker:
                            return None
                        return cached
                # 重试耗尽，直接调 loader
                logger.warning("锁等待超时, 直接加载: key=%s", raw_key)
                return loader()

        except redis.exceptions.RedisError:
            logger.warning("Redis 异常, 降级到直接加载: key=%s", raw_key)
            return loader()

        return loader()

    def get_stats(self) -> dict:
        """获取缓存统计"""
        r = self._ensure_redis()
        if r is None:
            return {"total": 0, "valid": 0, "expired": 0}
        try:
            info = r.info("keyspace")
            db_info = info.get("db0", {})
            keys = int(db_info.get("keys", 0))
        except redis.exceptions.RedisError:
            keys = 0
        return {
            "total": keys,
            "valid": keys,
            "expired": 0,
        }

    def clear(self) -> None:
        """清除所有缓存"""
        r = self._ensure_redis()
        if r is None:
            return
        try:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{self._key_prefix}*", count=100)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
        except redis.exceptions.RedisError:
            logger.warning("Redis 清空缓存失败", exc_info=True)

    # ----------------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def _jitter_ttl(self, base_ttl: int) -> int:
        """雪崩防护：TTL 随机抖动 ±20%"""
        pct = self._ttl_jitter_pct
        if pct <= 0:
            return base_ttl
        delta = int(base_ttl * pct / 100)
        return max(1, base_ttl + random.randint(-delta, delta))
