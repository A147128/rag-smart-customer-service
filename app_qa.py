import time
from rag_enhanced import EnhancedRagService
import streamlit as st
import config_data as config

# 页面配置
st.set_page_config(
    page_title="智能客服系统 - 增强版",
    page_icon="🤖",
    layout="wide"
)

# 标题
st.title("🤖 智能客服系统")
st.caption("基于RAG + 混合检索 + 响应缓存")
st.divider()

# 侧边栏 - 显示系统信息和缓存统计
with st.sidebar:
    st.header("⚙️ 系统信息")
    st.info("**技术栈:**\n- LangChain + Chroma\n- 混合检索(BM25+向量)\n- 响应缓存机制\n- 通义千问LLM")

    # 初始化或获取RAG服务
    if "rag" not in st.session_state:
        with st.spinner("初始化RAG服务..."):
            st.session_state["rag"] = EnhancedRagService(
                use_cache=True,
                use_hybrid_retrieval=True
            )
        st.success("✅ RAG服务已就绪")

    # 显示缓存统计
    if st.button("📊 刷新缓存统计"):
        stats = st.session_state["rag"].get_cache_stats()
        st.metric("缓存总数", stats.get('total', 0))
        st.metric("有效缓存", stats.get('valid', 0))
        st.metric("过期缓存", stats.get('expired', 0))

    # 清空缓存按钮
    if st.button("🗑️ 清空缓存"):
        result = st.session_state["rag"].clear_cache()
        st.success(result)
        st.rerun()

# 初始化消息历史
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好！我是智能客服助手，有什么可以帮助你的吗？"}]

# 显示历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # 显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # AI回复
    with st.chat_message("assistant"):
        ai_res_list = []
        start_time = time.time()

        with st.spinner("AI思考中..."):
            try:
                res_stream = st.session_state["rag"].chain.stream(
                    {"input": prompt},
                    config.session_config
                )

                def capture(generator, cache_list):
                    for chunk in generator:
                        cache_list.append(chunk)
                        yield chunk

                response = st.write_stream(capture(res_stream, ai_res_list))
                elapsed_time = time.time() - start_time

                # 显示响应时间
                st.caption(f"⏱️ 响应时间: {elapsed_time:.2f}秒")

            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")
                ai_res_list = [f"抱歉，处理您的问题时遇到错误: {str(e)}"]

    # 保存AI回复到历史
    st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

# ["a", "b", "c"]   "".join(list)    -> abc
# ["a", "b", "c"]   ",".join(list)    -> a,b,c