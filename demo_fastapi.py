"""
FastAPI 演示 - 简化版，快速体验
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 创建 FastAPI 应用
app = FastAPI(title="RAG 智能客服 API 演示", description="基于 FastAPI 的简单演示", version="1.0.0")


# 定义数据模型
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    answer: str
    source: str = "demo"


# API 路由
@app.get("/")
async def root():
    """根路径 - 欢迎信息"""
    return {"message": "欢迎使用 RAG 智能客服 API", "docs": "/docs", "api": "/api/chat"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "rag-api-demo"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口 - 模拟 RAG 回复
    """
    # 模拟 RAG 处理
    user_msg = request.message

    # 简单的关键词回复（实际项目会调用 RAG 链）
    if "天气" in user_msg:
        answer = "今天北京天气晴朗，25度，适合出行。"
    elif "你好" in user_msg:
        answer = "你好！我是智能客服助手，有什么可以帮你的吗？"
    elif "帮助" in user_msg:
        answer = "我可以回答关于产品、天气、时间的问题。请直接提问！"
    else:
        answer = f"收到你的问题：'{user_msg}'。在实际项目中，这里会调用 RAG 链进行检索和生成。"

    return {"answer": answer, "source": "RAG Demo"}


@app.get("/api/stats")
async def get_stats():
    """统计信息接口"""
    return {"total_requests": 100, "cache_hit_rate": "85%", "active_sessions": 5}


# 运行命令（在终端执行）：
# uvicorn demo_fastapi:app --reload --host 0.0.0.0 --port 8000
