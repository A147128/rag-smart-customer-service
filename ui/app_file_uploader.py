"""
基于Streamlit完成WEB网页上传服务

pip install streamlit

Streamlit：当WEB页面元素发生变化，则代码重新执行一遍

支持文件类型：txt, md, pdf, docx, xlsx, pptx
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import config_data as config
from service.knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")
st.markdown(f"""
**支持的文件类型**: txt, md, pdf, docx, xlsx, pptx

**限制条件**:
- 单文件大小 ≤ {config.MAX_FILE_SIZE_MB}MB
- 单次最多上传 {config.MAX_FILES_PER_UPLOAD} 个文件
- 文件编码仅支持 UTF-8（txt/md）
""")

# file_uploader - 支持多类型、多文件
uploader_files = st.file_uploader(
    "请上传知识库文件",
    type=list(config.ALLOWED_FILE_TYPES.keys()),
    accept_multiple_files=True,  # 支持多文件上传
)

# session_state就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


if uploader_files:
    # 检查文件数量
    if len(uploader_files) > config.MAX_FILES_PER_UPLOAD:
        st.error(f"单次最多上传 {config.MAX_FILES_PER_UPLOAD} 个文件，当前: {len(uploader_files)} 个")
    else:
        st.subheader(f"已选择 {len(uploader_files)} 个文件")

        # 显示文件信息
        for uploader_file in uploader_files:
            file_name = uploader_file.name
            file_type = uploader_file.type or "未知"
            file_size = format_file_size(uploader_file.size)

            # 获取文件扩展名
            ext = file_name.split(".")[-1].lower() if "." in file_name else "未知"

            with st.expander(f"📄 {file_name}", expanded=False):
                st.write(f"**类型**: {ext} | **MIME**: {file_type} | **大小**: {file_size}")

                # 检查文件大小限制
                if uploader_file.size > config.MAX_FILE_SIZE_BYTES:
                    st.warning(f"⚠️ 文件大小超过限制 ({config.MAX_FILE_SIZE_MB}MB)，将被跳过")

        # 上传按钮
        if st.button("上传到知识库", type="primary"):
            success_count = 0
            error_count = 0
            skipped_count = 0
            total_chunks = 0
            results_detail = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploader_file in enumerate(uploader_files):
                file_name = uploader_file.name
                progress_bar.progress((idx + 1) / len(uploader_files))
                status_text.text(f"正在处理: {file_name}...")

                # 检查文件大小
                if uploader_file.size > config.MAX_FILE_SIZE_BYTES:
                    results_detail.append({
                        "filename": file_name,
                        "status": "error",
                        "message": f"文件大小超过限制 ({config.MAX_FILE_SIZE_MB}MB)",
                    })
                    error_count += 1
                    continue

                try:
                    file_bytes = uploader_file.getvalue()
                    result = st.session_state["service"].upload_file(file_bytes, file_name)

                    results_detail.append({
                        "filename": file_name,
                        "status": result["status"],
                        "message": result["message"],
                        "chunks": result.get("chunks"),
                        "warnings": result.get("warnings"),
                    })

                    if result["status"] == "success":
                        success_count += 1
                        total_chunks += result.get("chunks", 0)
                    elif result["status"] == "skipped":
                        skipped_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    results_detail.append({
                        "filename": file_name,
                        "status": "error",
                        "message": f"处理失败: {str(e)}",
                    })
                    error_count += 1

            progress_bar.empty()
            status_text.empty()

            # 显示结果汇总
            st.subheader("上传结果汇总")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("成功", success_count)
            col2.metric("跳过", skipped_count)
            col3.metric("失败", error_count)
            col4.metric("总Chunks", total_chunks)

            # 显示详细结果
            st.subheader("详细结果")
            for detail in results_detail:
                status_icon = {
                    "success": "✅",
                    "skipped": "⏭️",
                    "error": "❌",
                }.get(detail["status"], "❓")

                with st.expander(f"{status_icon} {detail['filename']}", expanded=False):
                    st.write(f"**状态**: {detail['status']}")
                    st.write(f"**消息**: {detail['message']}")
                    if detail.get("chunks"):
                        st.write(f"**Chunks**: {detail['chunks']}")
                    if detail.get("warnings"):
                        st.warning(f"**警告**: {', '.join(detail['warnings'])}")

            # 显示知识库版本
            from service.knowledge_version import get_kb_version
            st.info(f"当前知识库版本: {get_kb_version()}")