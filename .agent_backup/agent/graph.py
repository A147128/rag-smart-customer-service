"""LangGraph StateGraph for outfit recommendation with HITL, parallel, and loop edges."""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.edges import handle_feedback, route_collect_requirements
from agent.nodes import (
    collect_requirements,
    create_order,
    generate_outfit,
    load_profile,
    parallel_retrieve,
    wait_feedback,
)
from agent.state import OutfitState


def _create_workflow() -> StateGraph:
    """Create the StateGraph with all nodes and edges (no checkpointer)."""
    workflow = StateGraph(OutfitState)

    workflow.add_node("collect_requirements", collect_requirements)
    workflow.add_node("load_profile", load_profile)
    workflow.add_node("parallel_retrieve", parallel_retrieve)
    workflow.add_node("generate_outfit", generate_outfit)
    workflow.add_node("wait_feedback", wait_feedback)
    workflow.add_node("create_order", create_order)

    workflow.add_edge(START, "collect_requirements")
    workflow.add_conditional_edges(
        "collect_requirements",
        route_collect_requirements,
        {"load_profile": "load_profile", "collect_requirements": "collect_requirements"},
    )
    workflow.add_edge("load_profile", "parallel_retrieve")
    workflow.add_edge("parallel_retrieve", "generate_outfit")
    workflow.add_edge("generate_outfit", "wait_feedback")
    workflow.add_conditional_edges(
        "wait_feedback",
        handle_feedback,
        {
            "create_order": "create_order",
            "generate_outfit": "generate_outfit",
            "__end__": END,
        },
    )
    workflow.add_edge("create_order", END)
    return workflow


def build_graph(db_path: str = "agent_checkpoints.db", checkpointer=None):
    """Build and compile the outfit recommendation graph.

    Args:
        db_path: Path to SQLite checkpoint file for state persistence.
        checkpointer: Optional pre-created checkpointer. If None, creates
                      SqliteSaver using db_path.

    Returns:
        Compiled LangGraph StateGraph.
    """
    workflow = _create_workflow()
    if checkpointer is None:
        import sqlite3

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)


async def build_async_graph(db_path: str = "agent_checkpoints.db"):
    """Build and compile with AsyncSqliteSaver for async streaming support.

    Uses the same SQLite database as build_graph(), so checkpoints
    are fully compatible between sync and async graphs.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    workflow = _create_workflow()
    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "build_async_graph"]
