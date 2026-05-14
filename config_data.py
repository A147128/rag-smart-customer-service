
# 数据目录（统一管理）
data_dir = "./data"

# MD5去重记录
md5_path = "./data/md5.text"

# Chroma向量数据库
collection_name = "rag"
persist_directory = "./data/chroma_db"

# 对话历史存储
chat_history_dir = "./data/chat_history"

# 响应缓存
cache_file = "./data/response_cache.json"

# 文本分割器
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000

# 检索参数
similarity_threshold = 1

# 模型配置
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# 会话配置
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
