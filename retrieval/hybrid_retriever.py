"""
Hybrid retrieval: BM25 keyword retrieval + vector retrieval with RRF fusion.

The public ``invoke`` method keeps LangChain-compatible behavior and returns
``list[Document]``. ``invoke_with_scores`` exposes scored results for source
citation and offline evaluation.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import jieba
from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalResult:
    """A document returned by retrieval with an explainable score."""

    document: Document
    score: float
    rank: int
    channels: tuple[str, ...]


class BM25Retriever:
    """Simple BM25 retriever for Chinese keyword matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.doc_freqs: defaultdict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents and rebuild the BM25 index."""
        self.documents = documents
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Chinese text and drop blank tokens."""
        return [token.strip() for token in jieba.cut(text) if token.strip()]

    def _build_index(self) -> None:
        """Build BM25 statistics from the current documents."""
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_lengths = []
        self.avg_doc_length = 0

        if not self.documents:
            return

        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        if self.avg_doc_length == 0:
            return

        n_docs = len(self.documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query: str) -> list[float]:
        """Compute BM25 scores for every indexed document."""
        if not self.documents or self.avg_doc_length == 0:
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = []

        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_tokens = self._tokenize(doc.page_content)
            doc_len = self.doc_lengths[idx]
            term_freq = Counter(doc_tokens)

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = term_freq.get(token, 0)
                if tf <= 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                score += self.idf[token] * (numerator / denominator)

            scores.append(score)

        return scores

    def invoke(self, query: str, k: int = 5) -> list[Document]:
        return [doc for doc, _score in self.invoke_with_scores(query, k=k)]

    def invoke_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if not self.documents:
            return []

        indexed_scores = list(enumerate(self.get_scores(query)))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in indexed_scores[:k] if score > 0]


class HybridRetriever:
    """Combine vector and BM25 retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def invoke(self, query: str, k: int = 5) -> list[Document]:
        return [result.document for result in self.invoke_with_scores(query, k=k)]

    def invoke_with_scores(self, query: str, k: int = 5) -> list[RetrievalResult]:
        fused: dict[str, dict[str, Any]] = {}

        self._accumulate_rrf(fused, self._vector_results(query), channel="vector")
        self._accumulate_rrf(
            fused,
            self.bm25_retriever.invoke_with_scores(query, k=max(self.candidate_k, k)),
            channel="bm25",
        )

        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        results: list[RetrievalResult] = []
        for rank, item in enumerate(ranked[:k], start=1):
            doc = item["document"]
            metadata = dict(doc.metadata or {})
            metadata.update(
                {
                    "retrieval_score": item["score"],
                    "retrieval_rank": rank,
                    "retrieval_channels": ",".join(sorted(item["channels"])),
                }
            )
            results.append(
                RetrievalResult(
                    document=Document(page_content=doc.page_content, metadata=metadata),
                    score=item["score"],
                    rank=rank,
                    channels=tuple(sorted(item["channels"])),
                )
            )
        return results

    def _vector_results(self, query: str) -> list[tuple[Document, float]]:
        raw_results = self.vector_retriever.invoke(query)
        normalized: list[tuple[Document, float]] = []
        for item in raw_results:
            if isinstance(item, tuple) and len(item) >= 2:
                doc, score = item[0], float(item[1])
            else:
                doc, score = item, 0.0
            normalized.append((doc, score))
        return normalized[: self.candidate_k]

    def _accumulate_rrf(
        self,
        fused: dict[str, dict[str, Any]],
        ranked_docs: list[tuple[Document, float]],
        channel: str,
    ) -> None:
        for rank, (doc, _score) in enumerate(ranked_docs, start=1):
            doc_id = self._get_doc_id(doc)
            item = fused.setdefault(doc_id, {"document": doc, "score": 0.0, "channels": set()})
            item["score"] += 1.0 / (self.rrf_k + rank)
            item["channels"].add(channel)

    def _get_doc_id(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        for key in ("content_md5", "chunk_id", "id"):
            if metadata.get(key):
                return str(metadata[key])
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
