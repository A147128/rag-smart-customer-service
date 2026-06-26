"""Conditional edges for outfit recommendation graph."""

from typing import Literal

from loguru import logger


def route_collect_requirements(state: dict) -> Literal["load_profile", "collect_requirements"]:
    missing = state.get("missing_info", [])
    occasion = state.get("occasion", "") or ""
    if occasion and not missing:
        logger.info("[route_collect] info complete, -> load_profile")
        return "load_profile"
    logger.info("[route_collect] info incomplete, -> collect_requirements")
    return "collect_requirements"


def handle_feedback(state: dict) -> Literal["create_order", "generate_outfit", "__end__"]:
    feedback = state.get("feedback", "").strip().lower()
    retry_count = state.get("retry_count", 0)

    # Check dissatisfaction keywords first (they take priority)
    unsatisfied = any(
        kw in feedback
        for kw in [
            "\u4e0d\u6ee1\u610f",
            "unsatisfied",
            "dislike",
            "no",
            "bad",
            "terrible",
        ]
    )
    if unsatisfied:
        if retry_count < 3:
            logger.info("[handle_feedback] unsatisfied (retry={}), -> generate_outfit", retry_count + 1)
            return "generate_outfit"
        logger.info("[handle_feedback] max retries, -> __end__ (transfer to human)")
        return "__end__"

    # Check satisfaction keywords
    satisfied = any(
        kw in feedback
        for kw in [
            "\u6ee1\u610f",
            "\u53ef\u4ee5",
            "\u4e0d\u9519",
            "\u597d\u7684",
            "satisfied",
            "accept",
            "yes",
            "ok",
            "good",
            "great",
        ]
    )
    if satisfied:
        logger.info("[handle_feedback] satisfied, -> create_order")
        return "create_order"

    # Ambiguous / unclear feedback: treat as unsatisfied but retry
    if retry_count < 3:
        logger.info("[handle_feedback] unclear feedback (retry={}), -> generate_outfit", retry_count + 1)
        return "generate_outfit"

    logger.info("[handle_feedback] unclear feedback & max retries, -> __end__")
    return "__end__"
