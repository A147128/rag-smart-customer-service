"""LangGraph nodes for outfit recommendation workflow."""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from loguru import logger

from agent.tools import check_inventory, knowledge_tool

_COLLECT_SYSTEM = (
    "你是服装推荐助手，需要收集用户的穿搭需求。\n"
    "规则：\n"
    "1. 仔细阅读对话历史，用户已经提供过的信息不要再问。\n"
    "2. 如果用户说的是全新话题（与之前讨论的衣服无关），说明用户切换了话题，应重新开始收集需求。\n"
    '3. 如果用户在反馈或追问上一轮的推荐（如"不够正式"、"换件上衣"等），说明场合已确定，应直接设置complete=true，occasion沿用已有值。\n'
    "4. 每次只问 1 个问题，逐步收集场合、季节、风格、预算等信息。\n"
    "5. 信息收集完整时（场合+季节+风格+预算），设置complete=true并填写occasion字段；否则设置complete=false并说明还缺少什么。\n"
    "6. 必须以JSON格式输出，不要包含其他内容。\n"
    'JSON格式：{"complete": true/false, "occasion": "场合", "missing": ["缺少信息"], "question": "追问的问题"}'
)

_MAX_COLLECT_LOOPS = 3


def _parse_json_response(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _trim_messages(messages: list, max_pairs: int = 2) -> str:
    """取最后 N 轮对话（1轮=user+assistant），返回文本。"""
    pairs = []
    current = []
    for msg in reversed(messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content or ""
        current.insert(0, f"{role}: {content}")
        if role == "user" and current:
            pairs.insert(0, "\n".join(current))
            current = []
            if len(pairs) >= max_pairs:
                break
    return "\n---\n".join(pairs)


def collect_requirements(state: dict) -> dict:
    from langchain_community.chat_models.tongyi import ChatTongyi

    from config import config_data as config

    messages = state.get("messages", [])
    collect_count = state.get("collect_count", 0)
    occasion = state.get("occasion", "") or ""

    # 兜底：超过最大轮次直接放过
    if collect_count >= _MAX_COLLECT_LOOPS:
        final_occasion = occasion or "日常穿搭"
        logger.warning("[collect_requirements] max loops, force proceed")
        return {"occasion": final_occasion, "missing_info": [], "collect_count": collect_count + 1}

    if not messages:
        return {"occasion": "", "missing_info": ["未知"], "collect_count": 1}

    # 如果最后一条是 AI 消息，说明在等待用户回答，中断等待用户输入
    if messages and isinstance(messages[-1], AIMessage):
        logger.info("[collect_requirements] waiting for user input, interrupt")
        response = interrupt(
            {
                "question": messages[-1].content,
                "prompt": "请回答以上问题",
            }
        )
        user_input = response if isinstance(response, str) else str(response)
        new_messages = list(messages) + [HumanMessage(content=user_input)]
        # 中断等待不计入 LLM 调用次数
        return {"messages": new_messages, "missing_info": ["待收集"]}

    # 只取最近 2 轮对话，避免旧方案干扰
    recent_text = _trim_messages(messages, max_pairs=2)

    llm = ChatTongyi(model=config.chat_model_name)
    response = llm.invoke(
        [
            SystemMessage(content=_COLLECT_SYSTEM),
            HumanMessage(
                content=f'最近对话：\n{recent_text}\n\n已有场合："{occasion}"\n请分析以上对话，提取穿搭需求，以JSON格式返回。'
            ),
        ]
    )

    result = _parse_json_response(response.content)
    if result is None:
        result = {"complete": False, "occasion": occasion, "missing": ["未知"], "question": "请问您有什么穿搭需求？"}

    # 增强校验：complete=true 但 occasion 为空 → 标记未完成
    if result.get("complete") and not result.get("occasion"):
        result["complete"] = False
        result.setdefault("missing", ["场合"])
        logger.warning("[collect_requirements] LLM返回complete=True但occasion为空，强制转为未完成")

    logger.info(
        "[collect_requirements] loop={}, complete={}, occasion={}",
        collect_count + 1,
        result.get("complete"),
        result.get("occasion"),
    )

    if result.get("complete"):
        return {"occasion": result["occasion"], "missing_info": [], "collect_count": collect_count + 1}

    question = result.get("question", "请问您有什么穿搭需求？")
    ai_msg = AIMessage(content=question or "请问您有什么穿搭需求？")
    new_messages = list(messages) + [ai_msg]
    return {
        "messages": new_messages,
        "missing_info": result.get("missing", ["未知"]),
        "occasion": result.get("occasion", occasion),
        "collect_count": collect_count + 1,
    }


def load_profile(state: dict) -> dict:
    if not state.get("profile"):
        logger.info("[load_profile] new user")
        return {"profile": "新用户"}
    return {"profile": state["profile"]}


def parallel_retrieve(state: dict) -> dict:
    occasion = state.get("occasion", "") or ""
    messages = state.get("messages", [])

    # 用完整的用户输入做查询，避免用 "??"
    user_query = occasion
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                user_query = msg.content
                break

    query = f"{occasion} {user_query}" if occasion and occasion not in ("??", "") else user_query
    logger.info("[parallel_retrieve] query={}", query)

    knowledge_result = knowledge_tool.invoke({"question": query})
    logger.info("[parallel_retrieve] knowledge done, len={}", len(knowledge_result))

    category = occasion or "general"
    inventory_result = check_inventory.invoke({"item_category": category})
    logger.info("[parallel_retrieve] inventory done")

    return {"knowledge_result": knowledge_result, "inventory_result": inventory_result}


_GENERATE_PROMPT = (
    "请根据以下信息生成穿搭推荐：\n"
    "场合：{occasion}\n"
    "用户画像：{profile}\n"
    "知识库参考：{knowledge_result}\n"
    "库存信息：{inventory_result}\n"
    "请给出具体、实用的搭配建议。"
)


def generate_outfit(state: dict) -> dict:
    from langchain_community.chat_models.tongyi import ChatTongyi

    from config import config_data as config

    prompt = _GENERATE_PROMPT.format(
        occasion=state.get("occasion", ""),
        profile=state.get("profile", "??"),
        knowledge_result=state.get("knowledge_result", ""),
        inventory_result=state.get("inventory_result", ""),
    )

    llm = ChatTongyi(model=config.chat_model_name)
    response = llm.invoke(
        [
            SystemMessage(content="你是专业的服装搭配师，请给出实用、具体的穿搭推荐。"),
            HumanMessage(content=prompt),
        ]
    )

    outfit_plan = response.content or "无法生成推荐，请稍后再试"
    logger.info("[generate_outfit] done, len={}", len(outfit_plan))

    retry_count = state.get("retry_count", 0) + 1
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=outfit_plan))

    return {"outfit_plan": outfit_plan, "messages": messages, "retry_count": retry_count}


def create_order(state: dict) -> dict:
    logger.info("[create_order] order confirmed")
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content="订单已确认，感谢您的购买！"))
    return {"order_confirmed": True, "messages": messages}


def wait_feedback(state: dict) -> dict:
    logger.info("[wait_feedback] pausing for user feedback")
    outfit_plan = state.get("outfit_plan", "") or "未知"
    response = interrupt(
        {
            "outfit_plan": outfit_plan,
            "prompt": "满意以上推荐吗？请反馈满意/不满意",
        }
    )
    feedback = response if isinstance(response, str) else str(response)
    logger.info("[wait_feedback] feedback: {}", feedback[:50])
    return {"feedback": feedback}
