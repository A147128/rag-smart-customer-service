"""
响应缓存模块 - 提升重复问题的响应速度
使用简单的字典缓存,生产环境可替换为Redis
"""
import hashlib
import json
import os
from datetime import datetime, timedelta


class ResponseCache:
    """基于文件的响应缓存系统"""

    def __init__(self, cache_file="./response_cache.json", ttl_hours=24):
        self.cache_file = cache_file
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = self._load_cache()

    def _load_cache(self):
        """从文件加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """保存缓存到文件"""
        os.makedirs(os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.', exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _generate_key(self, question: str) -> str:
        """生成问题的哈希键"""
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def get(self, question: str) -> str | None:
        """获取缓存的响应"""
        key = self._generate_key(question)
        if key in self.cache:
            cached_data = self.cache[key]
            # 检查是否过期
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cache_time < self.ttl:
                print(f"[缓存命中] 问题: {question[:30]}...")
                return cached_data['response']
            else:
                # 删除过期缓存
                del self.cache[key]
        return None

    def set(self, question: str, response: str):
        """设置缓存"""
        key = self._generate_key(question)
        self.cache[key] = {
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()
        print(f"[缓存写入] 问题: {question[:30]}...")

    def clear(self):
        """清空缓存"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total = len(self.cache)
        expired = 0
        now = datetime.now()
        for data in self.cache.values():
            cache_time = datetime.fromisoformat(data['timestamp'])
            if now - cache_time >= self.ttl:
                expired += 1
        return {
            'total': total,
            'valid': total - expired,
            'expired': expired
        }
