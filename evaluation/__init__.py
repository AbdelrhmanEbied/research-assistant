"""RAG evaluation package: retrieval metrics + LLM-as-judge scoring."""

from evaluation.dataset import EvalItem, load_dataset
from evaluation.retrieval_metrics import (
    average_metrics,
    evaluate_retrieval,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.runner import (
    QualityResult,
    RAGEvaluator,
    RetrievalResult,
    build_service,
    index_corpus,
)

__all__ = [
    "EvalItem",
    "load_dataset",
    "evaluate_retrieval",
    "hit_rate",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "average_metrics",
    "RAGEvaluator",
    "RetrievalResult",
    "QualityResult",
    "build_service",
    "index_corpus",
]
