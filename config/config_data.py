import os

from dotenv import load_dotenv

load_dotenv()

# 数据目录
data_dir = "./data"

# MD5校验文件路径
md5_path = "./data/md5.text"

# 知识库版本文件路径，用于缓存失效和chunk元数据
kb_version_path = "./data/kb_version.txt"

# Chroma数据库配置
collection_name = "rag"
persist_directory = "./data/chroma_db"

# 聊天历史目录
chat_history_dir = "./data/chat_history"

# 响应缓存文件 - 用于缓存相同的查询结果
cache_file = "./data/response_cache.json"

# Redis 连接配置信息
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_password = os.getenv("REDIS_PASSWORD", "")
redis_ttl = 86400  # 缓存过期时间24小时

# 构建 Redis URL时处理密码
_redis_pass_part = f":{redis_password}@" if redis_password else ""
redis_url = f"redis://{_redis_pass_part}{redis_host}:{redis_port}/0"

# MySQL 数据库配置
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
mysql_user = os.getenv("MYSQL_USER", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", "")
mysql_database = os.getenv("MYSQL_DATABASE", "chat_history")

# 文本分割配置
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "，", "；", " ", ""]
max_split_char_number = 1000

# 检索配置
retrieval_top_k = 5
retrieval_candidate_k = 20
rrf_k = 60
rerank_top_k = 5

# 兼容旧变量：原 similarity_threshold 实际被当作 top-k 使用
similarity_threshold = retrieval_top_k

# 模型配置
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen-plus"

# 默认会话配置
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}

# 文件上传配置
MAX_FILE_SIZE_MB = 10  # 单文件最大 10MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES_PER_UPLOAD = 5  # 单次最多上传 5 个文件

# 支持的文件类型及其 MIME 类型映射
ALLOWED_FILE_TYPES = {
    "txt": ["text/plain"],
    "md": ["text/markdown", "text/plain"],
    "pdf": ["application/pdf"],
    "docx": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    "xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    "pptx": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
}

# 文件扩展名到 doc_type 的映射
EXT_TO_DOC_TYPE = {
    ".txt": "txt",
    ".md": "md",
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
}

