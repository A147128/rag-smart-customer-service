from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import config_data as config
from service.knowledge_base import KnowledgeBaseService
from service.knowledge_version import get_kb_version
from service.rag_enhanced import EnhancedRagService


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class SourceItem(BaseModel):
    source: str
    preview: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    source: str = "unknown"
    rewritten_query: str = ""
    kb_version: int = 1
    sources: list[SourceItem] = Field(default_factory=list)


class UploadResult(BaseModel):
    filename: str
    status: str
    message: str
    chunks: int | None = None
    kb_version: int | None = None
    warnings: list[str] | None = None


class BatchUploadResponse(BaseModel):
    results: list[UploadResult]
    kb_version: int
    total_chunks: int
    success_count: int
    error_count: int
    skipped_count: int


rag_service: EnhancedRagService | None = None
kb_service: KnowledgeBaseService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service, kb_service
    rag_service = EnhancedRagService(use_cache=True, use_hybrid_retrieval=True)
    kb_service = KnowledgeBaseService()
    logger.info("✅ AI 服务就绪")
    yield


app = FastAPI(
    title="RAG Smart Customer Service API",
    description="智能客服后端接口 - 支持多类型文件上传 (txt, md, pdf, docx, xlsx, pptx)",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "欢迎使用 RAG 智能客服 API",
        "docs": "/docs",
        "supported_file_types": list(config.ALLOWED_FILE_TYPES.keys()),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-api", "kb_version": get_kb_version()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="服务未初始化")
    try:
        return rag_service.ask(request.message, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/api/upload", response_model=BatchUploadResponse)
async def upload_knowledge(files: list[UploadFile] = File(...)):
    """
    上传知识库文件（支持多类型、多文件）

    支持的文件类型: txt, md, pdf, docx, xlsx, pptx
    单文件大小限制: 10MB
    单次上传文件数量限制: 5 个
    """
    if not kb_service:
        raise HTTPException(status_code=500, detail="服务未初始化")

    # 检查文件数量
    if len(files) > config.MAX_FILES_PER_UPLOAD:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"单次最多上传 {config.MAX_FILES_PER_UPLOAD} 个文件，当前: {len(files)} 个"
            },
        )

    results = []
    total_chunks = 0
    success_count = 0
    error_count = 0
    skipped_count = 0

    for file in files:
        filename = file.filename or "unknown"

        try:
            file_bytes = await file.read()
            result = kb_service.upload_file(file_bytes, filename)

            upload_result = UploadResult(
                filename=filename,
                status=result["status"],
                message=result["message"],
                chunks=result.get("chunks"),
                kb_version=result.get("kb_version"),
                warnings=result.get("warnings"),
            )
            results.append(upload_result)

            if result["status"] == "success":
                success_count += 1
                total_chunks += result.get("chunks", 0)
            elif result["status"] == "skipped":
                skipped_count += 1
            else:
                error_count += 1

        except Exception as e:
            logger.error(f"上传文件失败: {filename}, {e}")
            results.append(
                UploadResult(
                    filename=filename,
                    status="error",
                    message=f"上传失败: {str(e)}",
                )
            )
            error_count += 1

    return BatchUploadResponse(
        results=results,
        kb_version=get_kb_version(),
        total_chunks=total_chunks,
        success_count=success_count,
        error_count=error_count,
        skipped_count=skipped_count,
    )


@app.post("/api/upload/single")
async def upload_single_file(file: UploadFile = File(...)):
    """
    上传单个知识库文件（简化接口）

    支持的文件类型: txt, md, pdf, docx, xlsx, pptx
    """
    if not kb_service:
        raise HTTPException(status_code=500, detail="服务未初始化")

    filename = file.filename or "unknown"

    try:
        file_bytes = await file.read()
        result = kb_service.upload_file(file_bytes, filename)

        if result["status"] == "error":
            return JSONResponse(status_code=400, content={"detail": result["message"]})

        return {
            "status": result["status"],
            "message": result["message"],
            "filename": filename,
            "chunks": result.get("chunks"),
            "kb_version": get_kb_version(),
            "warnings": result.get("warnings"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    if rag_service and rag_service.use_cache:
        return rag_service.get_cache_stats()
    return {"enabled": False, "kb_version": get_kb_version()}