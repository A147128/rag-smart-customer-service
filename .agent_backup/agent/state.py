"""
LangGraph state definition for outfit recommendation
"""

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class OutfitState(TypedDict):
    """State for the outfit recommendation graph.

    Fields:
        messages:          Conversation history (LangChain BaseMessage list).
        profile:           User profile / persona description.
        occasion:          Wearing occasion extracted from user input.
        missing_info:      Missing information points identified during collection.
        knowledge_result:  Fashion knowledge retrieved by knowledge_tool.
        inventory_result:  Inventory status checked by inventory_tool.
        outfit_plan:       Final structured outfit recommendation.
        feedback:          User feedback on the recommendation.
        retry_count:       Number of regeneration retries (max 3).
        order_confirmed:   Whether the order has been created.
        needs_human:       Whether the conversation should be transferred to human agent.
    """

    messages: list[BaseMessage]
    profile: str
    occasion: str
    missing_info: list[str]
    knowledge_result: str
    inventory_result: str
    outfit_plan: str
    feedback: str
    collect_count: int
    retry_count: int
    order_confirmed: bool
    needs_human: bool
