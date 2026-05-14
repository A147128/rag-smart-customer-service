"""
增强版RAG服务 - 添加缓存和混合检索功能
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from cache_store import ResponseCache
from hybrid_retriever import BM25Retriever, HybridRetriever


def print_prompt(prompt):
    """调试用:打印prompt内容"""
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class EnhancedRagService(object):
    """增强版RAG服务,包含缓存和混合检索"""

    def __init__(self, use_cache=True, use_hybrid_retrieval=True):
        self.use_cache = use_cache
        self.use_hybrid_retrieval = use_hybrid_retrieval

        # 初始化向量服务
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 设置默认检索器(确保始终有retriever属性)
        self.retriever = self.vector_service.get_retriever()

        # 初始化缓存
        if self.use_cache:
            self.cache = ResponseCache(cache_file=config.cache_file, ttl_hours=24)
            print("[初始化] 响应缓存已启用")

        # 初始化混合检索器
        if self.use_hybrid_retrieval:
            try:
                self._init_hybrid_retriever()
                print("[初始化] 混合检索已启用")
            except Exception as e:
                print(f"[警告] 混合检索初始化失败: {e}")
                self.retriever = self.vector_service.get_retriever()

        # 初始化Prompt模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        # 初始化聊天模型
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 构建执行链
        self.chain = self.__get_chain()

    def _init_hybrid_retriever(self):
        """初始化混合检索器"""
        # 从向量库获取所有文档用于BM25索引
        vector_store = self.vector_service.vector_store
        all_docs = vector_store.get()

        if all_docs and 'documents' in all_docs:
            # 构建BM25检索器
            bm25_retriever = BM25Retriever(k1=1.5, b=0.75)
            documents = [
                Document(page_content=doc, metadata=all_docs['metadatas'][i])
                for i, doc in enumerate(all_docs['documents'])
            ]
            bm25_retriever.add_documents(documents)

            # 创建混合检索器
            vector_retriever = self.vector_service.get_retriever()
            self.retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                weights=(0.7, 0.3)
            )
            print(f"[初始化] BM25索引已构建,共{len(documents)}个文档")
        else:
            print("[警告] 向量库为空,使用纯向量检索")
            # self.retriever 已经在__init__中设置了

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.retriever

        def format_document(docs):
            """格式化检索结果"""
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for i, doc in enumerate(docs):
                formatted_str += f"参考资料{i + 1}: {doc.page_content}\n"
                formatted_str += f"来源: {doc.metadata.get('source', '未知')}\n\n"

            return formatted_str

        # 构建基础链
        # 用 RunnableLambda 包装自定义 retriever，兼容新版本 LangChain 的管道语法
        wrapped_retriever = RunnableLambda(lambda q: retriever.invoke(q))

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: x["input"]) | wrapped_retriever | RunnableLambda(format_document)
            } | RunnableLambda(lambda x: {
                "input": x["input"]["input"],
                "context": x["context"],
                "history": x["input"]["history"]
            }) | self.prompt_template | RunnableLambda(print_prompt) | self.chat_model | StrOutputParser()
        )

        # 如果启用缓存,包装缓存层
        if self.use_cache:
            chain = self._wrap_with_cache(chain)

        # 添加对话历史管理
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    def _wrap_with_cache(self, base_chain):
        """为链添加缓存包装"""
        def cached_invoke(input_dict, config=None):
            """带缓存的调用"""
            question = input_dict.get("input", "")

            # 尝试从缓存获取
            cached_response = self.cache.get(question)
            if cached_response:
                return cached_response

            # 缓存未命中,执行原始链
            response = base_chain.invoke(input_dict, config)

            # 写入缓存
            self.cache.set(question, response)

            return response

        return RunnableLambda(cached_invoke)

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.use_cache:
            return self.cache.get_stats()
        return {"enabled": False}

    def clear_cache(self):
        """清空缓存"""
        if self.use_cache:
            self.cache.clear()
            return "缓存已清空"
        return "缓存未启用"


# 保持向后兼容,原来的RagService指向增强版
RagService = EnhancedRagService
