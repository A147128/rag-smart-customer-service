"""LangChain Tools wrapping existing EnhancedRagService and inventory mock."""

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field

from service.rag_enhanced import EnhancedRagService

_rag_service: EnhancedRagService | None = None


def _get_rag() -> EnhancedRagService:
    global _rag_service
    if _rag_service is None:
        logger.info("[KnowledgeTool] RAG service init (singleton)")
        _rag_service = EnhancedRagService(use_cache=True, use_hybrid_retrieval=True)
    return _rag_service


class KnowledgeToolInput(BaseModel):
    question: str = Field(description="User question about outfits")
    session_id: str = Field(default="user_001", description="Session identifier")


@tool(args_schema=KnowledgeToolInput)
def knowledge_tool(question: str, session_id: str = "user_001") -> str:
    """Answer user questions based on fashion knowledge base with vector search and chat history."""
    rag = _get_rag()
    chain_config = {"configurable": {"session_id": session_id}}
    logger.info("[KnowledgeTool] invoke: question={}", question)
    return rag.chain.invoke({"input": question}, chain_config)


class InventoryInput(BaseModel):
    item_category: str = Field(description="Clothing category, e.g. suit/dress/casual/shirt/skirt")


@tool(args_schema=InventoryInput)
def check_inventory(item_category: str) -> str:
    """Check inventory status for a clothing category."""
    logger.info("[InventoryTool] check: category={}", item_category)
    _mock_db = {
        "suit": "Black suit in stock (M/L/XL), navy only L left, grey needs 2-3 days",
        "dress": "Rental dresses tight schedule (book 3 days ahead), purchase available black/navy full sizes",
        "casual": "T-shirts/hoodies abundant, jeans popular sizes limited",
        "shirt": "White/light blue shirts in stock, French cuff needs pre-order",
        "skirt": "Dress skirt M/L available, half skirt S out of stock",
        "shoes": "Leather shoes sizes 42/43 available, sneakers hot models sold out",
        "accessory": "Ties/belts in stock, cufflinks limited selection",
    }
    for key, value in _mock_db.items():
        if key in item_category.lower():
            return value
    return "No inventory info for this category, please contact customer service"
