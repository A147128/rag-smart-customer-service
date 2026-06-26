"""
Chat domain models
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request"""

    message: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    """Chat response"""

    answer: str
    source: str = "unknown"
