from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from evaluation.dataset import EvalItem
from rag.rag_schemas import RetrievedDocuments

Metrics = dict[str, float]


def _top_n(ranked: Sequence[str], k: int | None) -> list[str]:
    return list(ranked[:k]) if k is not None else list(ranked)


def hit_rate(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """1.0 if any relevant item appears in the top ``k``, else 0.0."""
    return 1.0 if any(rid in relevant for rid in _top_n(ranked, k)) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Fraction of relevant items retrieved in the top ``k``."""
    if not relevant:
        return 0.0
    top = _top_n(ranked, k)
    return sum(1 for rid in top if rid in relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Fraction of the top ``k`` items that are relevant."""
    top = _top_n(ranked, k)
    if not top:
        return 0.0
    return sum(1 for rid in top if rid in relevant) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item, 0.0 if none is retrieved."""
    for rank, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Normalized DCG with binary gains on the top ``k`` items."""
    if not relevant:
        return 0.0
    top = _top_n(ranked, k)
    dcg = sum(
        1.0 / math.log2(rank + 1.0) for rank, rid in enumerate(top, start=1) if rid in relevant
    )
    max_relevant = min(len(top), len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, max_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _chunk_ids(documents: Sequence[RetrievedDocuments]) -> list[str]:
    return [str(doc.metadata.get("chunk_id")) for doc in documents]


def _document_ids(documents: Sequence[RetrievedDocuments]) -> list[str]:
    """Unique document ids in retrieval order.

    A document may contribute several chunks to the ranking; for document-level
    metrics only its first occurrence matters, otherwise duplicates inflate
    DCG beyond the ideal and skew precision.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for doc in documents:
        document_id = str(doc.metadata.get("document_id"))
        if document_id not in seen:
            seen.add(document_id)
            ids.append(document_id)
    return ids


def evaluate_retrieval(
    documents: Sequence[RetrievedDocuments],
    item: EvalItem,
    k: int,
) -> tuple[Metrics | None, Metrics | None]:
    """Score the retrieved ``documents`` against one dataset item.

    Returns ``(chunk_metrics, document_metrics)``; each is ``None`` when the
    item does not carry that ground-truth type.
    """
    chunk_metrics = None
    if item.relevant_chunk_ids:
        ranked = _chunk_ids(documents)
        relevant = set(item.relevant_chunk_ids)
        chunk_metrics = {
            "hit_rate": hit_rate(ranked, relevant, k),
            "recall": recall_at_k(ranked, relevant, k),
            "precision": precision_at_k(ranked, relevant, k),
            "mrr": reciprocal_rank(ranked, relevant),
            "ndcg": ndcg_at_k(ranked, relevant, k),
        }

    document_metrics = None
    if item.relevant_document_ids:
        ranked = _document_ids(documents)
        relevant = set(item.relevant_document_ids)
        document_metrics = {
            "hit_rate": hit_rate(ranked, relevant, k),
            "recall": recall_at_k(ranked, relevant, k),
            "precision": precision_at_k(ranked, relevant, k),
            "mrr": reciprocal_rank(ranked, relevant),
            "ndcg": ndcg_at_k(ranked, relevant, k),
        }

    return chunk_metrics, document_metrics


def average_metrics(rows: Iterable[Metrics]) -> Metrics | None:
    """Mean of each metric across ``rows`` (None when there are no rows)."""
    rows = [row for row in rows if row]
    if not rows:
        return None
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
