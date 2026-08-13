from collections.abc import Sequence
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.rag_schemas import RetrievedDocuments
from telemetry import get_current_tracker


class Reranker:
    def __init__(
        self,
        model: str,
    ):
        cache_dir = Path.home() / ".cache" / "fastembed"
        self.model = TextCrossEncoder(model, cache_dir=str(cache_dir))

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocuments],
        top_k: int = 5,
    ) -> list[RetrievedDocuments]:

        with get_current_tracker().span(
            "rerank",
            span_type="RERANKER",
            latency_metric="reranker_latency_ms",
        ):
            if not documents:
                return []

            topk = min(top_k, len(documents))

            document_texts = [doc.text for doc in documents]

            results = list(
                self.model.rerank(
                    query=query,
                    documents=document_texts,
                )
            )

            ranked = sorted(
                zip(documents, results),
                key=lambda x: x[1],
                reverse=True,
            )

            return [doc for doc, _ in ranked[:topk]]
