from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.agent_schemas import PromptMode
from agent.llms import extract_llm_text, get_llms
from evaluation.dataset import EvalItem
from evaluation.judges import score_answer_relevance, score_context_relevance, score_faithfulness
from evaluation.retrieval_metrics import evaluate_retrieval
from rag.rag_schemas import SearchType
from rag.rag_service import RAGService, create_rag_service


class _IdentityReranker:
    """No-op reranker: keeps retrieval order, used when reranking is disabled.

    ``prepare`` already skips reranking when ``rerank=False``, but
    ``create_rag_service`` requires a reranker instance, so an identity object
    is passed instead of loading the cross-encoder model.
    """

    def rerank(self, query, documents, top_k=5):
        return list(documents)[:top_k]

    def rerank_diversified(self, query, documents, top_k=5):
        return list(documents)[:top_k]


def build_service(db_path: str, rerank: bool) -> RAGService:
    if rerank:
        from rag.reranker import Reranker

        reranker: Any = Reranker(model="Xenova/ms-marco-MiniLM-L-12-v2")
    else:
        reranker = _IdentityReranker()

    return create_rag_service(reranker=reranker, db_path=db_path)


def index_corpus(service: RAGService, corpus_dir: str | Path) -> None:
    """Index every supported file in ``corpus_dir`` with a per-file document_id."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    for path in sorted(root.iterdir()):
        if path.is_file():
            service.index(
                str(path),
                {"document_id": path.stem, "name": path.name},
            )


@dataclass
class RetrievalResult:
    search_type: str
    rerank: bool
    query: str
    retrieved_count: int
    chunk_metrics: dict[str, float] | None
    document_metrics: dict[str, float] | None


@dataclass
class QualityResult:
    query: str
    answer: str
    faithfulness: float | None
    answer_relevance: float | None
    context_relevance: float | None


class RAGEvaluator:
    def __init__(self, service: RAGService, k: int = 5, generation_llm=None):
        self.service = service
        self.k = k
        self.generation_llm = generation_llm

    def evaluate_retrieval(
        self,
        items: list[EvalItem],
        search_types: list[str],
        rerank: bool,
    ) -> list[RetrievalResult]:
        """Run retrieval for every item x search type and score against ground truth."""
        results: list[RetrievalResult] = []
        for item in items:
            for search_type in search_types:
                knowledge = self.service.prepare(
                    query=item.query,
                    mode=PromptMode.CHAT,
                    history=[],
                    retrieve=True,
                    rerank=rerank,
                    limit=self.k,
                    search_type=SearchType(search_type),
                    conversation_id=item.conversation_id,
                )
                ranked = knowledge.reranked_documents
                chunk_metrics, document_metrics = evaluate_retrieval(ranked, item, self.k)
                results.append(
                    RetrievalResult(
                        search_type=search_type,
                        rerank=rerank,
                        query=item.query,
                        retrieved_count=len(ranked),
                        chunk_metrics=chunk_metrics,
                        document_metrics=document_metrics,
                    )
                )
        return results

    def evaluate_quality(
        self,
        items: list[EvalItem],
        search_type: str,
        rerank: bool,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        api_key: str | None = None,
    ) -> list[QualityResult]:
        """Generate an answer and run LLM-as-judge metrics for every item."""
        if self.generation_llm is None:
            generation_llm = get_llms(model=model, model_provider=model_provider, api_key=api_key)[
                0
            ]
        else:
            generation_llm = self.generation_llm

        results: list[QualityResult] = []
        for item in items:
            knowledge = self.service.prepare(
                query=item.query,
                mode=PromptMode.CHAT,
                history=[],
                retrieve=True,
                rerank=rerank,
                limit=self.k,
                search_type=SearchType(search_type),
                conversation_id=item.conversation_id,
            )

            answer = extract_llm_text(generation_llm.invoke(knowledge.prompt))
            faithfulness = score_faithfulness(
                item.query,
                knowledge.context.text,
                answer,
                model=model,
                model_provider=model_provider,
                api_key=api_key,
            )
            answer_relevance = score_answer_relevance(
                item.query,
                answer,
                model=model,
                model_provider=model_provider,
                api_key=api_key,
            )
            context_relevance = score_context_relevance(
                item.query,
                knowledge.context.text,
                model=model,
                model_provider=model_provider,
                api_key=api_key,
            )

            results.append(
                QualityResult(
                    query=item.query,
                    answer=answer,
                    faithfulness=faithfulness.score if faithfulness else None,
                    answer_relevance=answer_relevance.score if answer_relevance else None,
                    context_relevance=context_relevance.score if context_relevance else None,
                )
            )
        return results
