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

    def _score(
        self,
        query: str,
        documents: Sequence[RetrievedDocuments],
    ) -> list[tuple[RetrievedDocuments, float]]:
        document_texts = [doc.text for doc in documents]
        results = list(
            self.model.rerank(
                query=query,
                documents=document_texts,
            )
        )
        return list(zip(documents, results, strict=True))

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

            ranked = sorted(
                self._score(query, documents),
                key=lambda x: x[1],
                reverse=True,
            )

            return [doc for doc, _ in ranked[:topk]]

    def rerank_diversified(
        self,
        query: str,
        documents: Sequence[RetrievedDocuments],
        top_k: int = 5,
    ) -> list[RetrievedDocuments]:
        """Rerank so every distinct document contributes before relevance fills.

        Walks the relevance-ranked list once, greedily taking the top chunk of
        each distinct ``document_id``, then fills any remaining slots in pure
        relevance order. With a single document this degenerates to ``rerank``.
        """
        with get_current_tracker().span(
            "rerank_diversified",
            span_type="RERANKER",
            latency_metric="reranker_latency_ms",
        ):
            if not documents:
                return []

            topk = min(top_k, len(documents))

            ranked = sorted(
                self._score(query, documents),
                key=lambda x: x[1],
                reverse=True,
            )

            selected: list[RetrievedDocuments] = []
            seen: set[str | None] = set()

            for doc, _ in ranked:
                if len(selected) >= topk:
                    break
                document_id = doc.metadata.get("document_id")
                if document_id not in seen:
                    seen.add(document_id)
                    selected.append(doc)

            for doc, _ in ranked:
                if len(selected) >= topk:
                    break
                if doc not in selected:
                    selected.append(doc)

            return selected
