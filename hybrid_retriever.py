"""
混合检索模块 - 结合关键词检索(BM25)和向量检索
提高检索准确率,特别是对于专有名词和精确匹配场景
"""
import jieba
from typing import List
from langchain_core.documents import Document
from collections import Counter
import math


class BM25Retriever:
    """基于BM25算法的关键词检索器"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # BM25参数
        self.b = b    # BM25参数
        self.documents = []
        self.doc_freqs = Counter()  # 词项文档频率
        self.idf = {}  # 逆文档频率
        self.doc_lengths = []  # 文档长度
        self.avg_doc_length = 0  # 平均文档长度

    def add_documents(self, documents: List[Document]):
        """添加文档到索引"""
        self.documents = documents
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def _build_index(self):
        """构建BM25索引"""
        if not self.documents:
            return

        # 统计文档频率和文档长度
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # 计算IDF
        n_docs = len(self.documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query: str) -> List[float]:
        """计算查询与所有文档的BM25分数"""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_tokens = self._tokenize(doc.page_content)
            doc_len = self.doc_lengths[idx]

            # 统计词频
            term_freq = Counter(doc_tokens)

            for token in query_tokens:
                if token in self.idf:
                    tf = term_freq.get(token, 0)
                    idf = self.idf[token]

                    # BM25公式
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)

            scores.append(score)

        return scores

    def invoke(self, query: str, k: int = 5) -> List[Document]:
        """检索相关文档"""
        if not self.documents:
            return []

        scores = self.get_scores(query)

        # 按分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top-k文档
        top_indices = [idx for idx, score in indexed_scores[:k] if score > 0]
        return [self.documents[idx] for idx in top_indices]


class HybridRetriever:
    """混合检索器 - 结合向量检索和BM25关键词检索"""

    def __init__(self, vector_retriever, bm25_retriever, weights=(0.7, 0.3)):
        """
        初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25关键词检索器
            weights: (向量权重, BM25权重),默认(0.7, 0.3)
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = weights[0]
        self.bm25_weight = weights[1]

    def invoke(self, query: str, k: int = 5) -> List[Document]:
        """
        执行混合检索

        Args:
            query: 查询文本
            k: 返回文档数量

        Returns:
            排序后的文档列表
        """
        # 1. 向量检索
        vector_docs = self.vector_retriever.invoke(query)

        # 2. BM25检索
        bm25_docs = self.bm25_retriever.invoke(query, k=k * 2)

        # 3. 合并结果(去重)
        all_docs = {}

        # 添加向量检索结果(赋予较高初始分数)
        for i, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # 向量检索排名越靠前分数越高
            score = self.vector_weight * (1.0 / (i + 1))
            all_docs[doc_id] = (doc, score)

        # 添加BM25检索结果
        for i, doc in enumerate(bm25_docs):
            doc_id = self._get_doc_id(doc)
            # BM25分数归一化
            bm25_score = self.bm25_weight * (1.0 / (i + 1))

            if doc_id in all_docs:
                # 如果已存在,累加分数
                existing_doc, existing_score = all_docs[doc_id]
                all_docs[doc_id] = (existing_doc, existing_score + bm25_score)
            else:
                all_docs[doc_id] = (doc, bm25_score)

        # 4. 按综合分数排序
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)

        # 返回top-k文档
        return [doc for doc, score in sorted_docs[:k]]

    def _get_doc_id(self, doc: Document) -> str:
        """生成文档唯一标识"""
        # 使用内容哈希作为ID
        import hashlib
        return hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
