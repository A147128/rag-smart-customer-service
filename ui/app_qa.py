# ruff: noqa: E402
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from config import config_data as config
from service.rag_enhanced import EnhancedRagService, _status_cb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="智能客服系统", page_icon="🎻", layout="wide")

st.title("🎻 智能客服系统")
st.caption("基于 RAG 的智能客服助手")
st.divider()

# ---------------------------------------------------------------------------
# Initialize messages
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是智能客服助手，有什么可以帮助你的吗？"}
    ]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 系统信息")
    st.info("**技术栈:**\n- LangChain + Chroma\n- 混合检索 (BM25+向量)\n- 响应缓存机制\n- 通义千问 LLM")

    if "rag" not in st.session_state:
        with st.spinner("初始化 RAG 服务..."):
            st.session_state["rag"] = EnhancedRagService(use_cache=True, use_hybrid_retrieval=True)
        st.success("✅ RAG 服务已就绪")

    if st.button("📳 刷新缓存统计"):
        stats = st.session_state["rag"].get_cache_stats()
        st.metric("缓存总数", stats.get("total", 0))
        st.metric("有效缓存", stats.get("valid", 0))
        st.metric("过期缓存", stats.get("expired", 0))

    if st.button("🗑️ 清空缓存"):
        result = st.session_state["rag"].clear_cache()
        st.success(result)
        st.rerun()

# ---------------------------------------------------------------------------
# Message display
# ---------------------------------------------------------------------------
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # AI response
    with st.chat_message("assistant"):
        status_box = st.status("思考中...", expanded=True)
        ai_res_list = []
        start_time = time.time()

        try:
            rag = st.session_state["rag"]
            _status_cb.set(lambda s: status_box.update(label=s))
            res_stream = rag.chain.stream({"input": prompt}, config.session_config)

            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)
                    yield chunk

            response = st.write_stream(capture(res_stream, ai_res_list))
            _status_cb.set(None)
            elapsed = time.time() - start_time
            status_box.update(label="完成", state="complete")
            st.caption(f"⚡ 响应时间: {elapsed:.2f}秒")

        except Exception as e:
            _status_cb.set(None)
            status_box.update(label="错误", state="error")
            st.error(f"❌ 发生错误: {str(e)}")
            ai_res_list = [f"抱歉，处理您的问题时遇到错误: {str(e)}"]

    st.session_state["messages"].append({"role": "assistant", "content": "".join(ai_res_list)})
    st.rerun()