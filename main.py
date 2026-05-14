"""
智能客服系统 - FastAPI 后端服务
基于 RAG + 通义千问 + Chroma 向量库
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 导入核心服务
from rag_enhanced import EnhancedRagService
from knowledge_base import KnowledgeBaseService
import config_data as config

# 定义数据模型 (Pydantic)
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"

class ChatResponse(BaseModel):
    answer: str
    source: str = "unknown"

# 全局服务实例 (避免每次请求都初始化)
rag_service: EnhancedRagService = None
kb_service: KnowledgeBaseService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    global rag_service, kb_service
    
    print("🚀 正在初始化 AI 服务...")
    # 懒加载服务，避免启动超时
    rag_service = EnhancedRagService(use_cache=True, use_hybrid_retrieval=True)
    kb_service = KnowledgeBaseService()
    print("✅ AI 服务初始化完成")
    
    yield

# 初始化 FastAPI
app = FastAPI(
    title="RAG Smart Customer Service API",
    description="基于 LangChain 和通义千问的智能客服后端接口",
    version="1.0.0",
    lifespan=lifespan
)

# --- API 路由 ---

@app.get("/")
async def root():
    return {"message": "欢迎使用 RAG 智能客服 API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-api"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    try:
        # 调用 RAG 服务
        # 注意：这里使用 invoke 而非 stream，方便 API 返回完整结果
        session_config = {"configurable": {"session_id": request.session_id}}
        response = rag_service.chain.invoke(
            {"input": request.message}, 
            session_config
        )
        
        return {
            "answer": response,
            "source": "RAG Knowledge Base"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.post("/api/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """
    知识库上传接口
    """
    if not kb_service:
        raise HTTPException(status_code=500, detail="服务未初始化")
        
    # 检查文件类型
    if not file.filename.endswith('.txt'):
        return JSONResponse(status_code=400, content={"detail": "仅支持 .txt 文件"})
    
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        # 存入知识库
        result = kb_service.upload_by_str(text, file.filename)
        return {"message": result, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """
    获取缓存统计信息
    """
    if rag_service and rag_service.use_cache:
        return rag_service.get_cache_stats()
    return {"enabled": False}
