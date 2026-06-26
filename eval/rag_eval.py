"""Offline RAG retrieval evaluation.

Run:
    python eval/rag_eval.py

Metrics:
    Recall@K: whether any expected source appears in top-k.
    MRR: reciprocal rank of the first expected source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config_data as config
from service.rag_enhanced import EnhancedRagService


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(dataset: list[dict], top_k: int = 5) -> dict:
    service = EnhancedRagService(use_cache=False, use_hybrid_retrieval=True)
    hits = 0
    reciprocal_ranks = []
    details = []

    for item in dataset:
        query = item["query"]
        expected = set(item.get("expected_sources", []))
        results = service.retrieve_documents(query)[:top_k]
        sources = [str(result.document.metadata.get("source", "")) for result in results]

        first_rank = 0
        for idx, source in enumerate(sources, start=1):
            if source in expected:
                first_rank = idx
                break

        if first_rank:
            hits += 1
            reciprocal_ranks.append(1 / first_rank)
        else:
            reciprocal_ranks.append(0)

        details.append({"query": query, "expected": sorted(expected), "sources": sources, "rank": first_rank})

    total = len(dataset) or 1
    return {
        "top_k": top_k,
        "recall_at_k": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "details": details,
    }


def main() -> None:
    dataset_path = Path(__file__).with_name("rag_eval_dataset.json")
    report = evaluate(load_dataset(dataset_path), top_k=config.retrieval_top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
