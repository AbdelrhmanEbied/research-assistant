import math
from types import SimpleNamespace

import pytest

from evaluation.dataset import EvalItem, load_dataset
from evaluation.report import build_report, render_markdown
from evaluation.retrieval_metrics import (
    average_metrics,
    evaluate_retrieval,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.runner import RAGEvaluator, RetrievalResult
from rag.rag_schemas import RetrievedDocuments


def make_doc(document_id, chunk_id, text="content", score=0.5):
    return RetrievedDocuments(
        text=text,
        metadata={"document_id": document_id, "chunk_id": chunk_id},
        score=score,
    )


# --- metric correctness ----------------------------------------------


def test_hit_rate():
    ranked = ["a", "b", "c"]
    assert hit_rate(ranked, {"b"}, k=3) == 1.0
    assert hit_rate(ranked, {"b"}, k=1) == 0.0
    assert hit_rate(ranked, {"z"}, k=3) == 0.0


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"a", "c", "d"}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, {"z"}, k=3) == 0.0
    assert recall_at_k(ranked, set(), k=3) == 0.0


def test_precision_at_k():
    ranked = ["a", "b", "c"]
    assert precision_at_k(ranked, {"a", "c"}, k=2) == pytest.approx(0.5)
    assert precision_at_k(ranked, {"a", "c"}, k=3) == pytest.approx(2 / 3)
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "y"], {"z"}) == 0.0


def test_ndcg_at_k():
    ranked = ["a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(dcg / idcg)
    # perfectly ranked → 1.0
    assert ndcg_at_k(["a", "c"], relevant, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, set(), k=3) == 0.0


# --- evaluate_retrieval ----------------------------------------------


def test_evaluate_retrieval_chunk_and_document_metrics():
    docs = [
        make_doc("d1", "c1", score=0.9),
        make_doc("d2", "c2", score=0.8),
        make_doc("d1", "c3", score=0.7),
    ]
    item = EvalItem(query="q", relevant_document_ids=["d1"], relevant_chunk_ids=["c1", "c3"])

    chunk, document = evaluate_retrieval(docs, item, k=3)

    assert chunk["hit_rate"] == 1.0
    assert chunk["recall"] == 1.0
    assert chunk["precision"] == pytest.approx(2 / 3)
    assert chunk["mrr"] == 1.0
    # ranked c1, c2, c3 with relevant {c1, c3}
    assert chunk["ndcg"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))

    # document ids are deduplicated → d1, d2
    assert document["hit_rate"] == 1.0
    assert document["recall"] == 1.0
    assert document["precision"] == pytest.approx(0.5)
    assert document["mrr"] == 1.0
    assert document["ndcg"] == pytest.approx(1.0)


def test_evaluate_retrieval_missing_ground_truth_is_none():
    docs = [make_doc("d1", "c1")]
    item = EvalItem(query="q")
    chunk, document = evaluate_retrieval(docs, item, k=3)
    assert chunk is None
    assert document is None


def test_average_metrics():
    rows = [{"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.5}]
    assert average_metrics(rows) == {"a": 0.5, "b": 0.5}
    assert average_metrics([]) is None


# --- dataset loader ----------------------------------------------------


def test_load_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"query": "q1", "relevant_document_ids": ["d1"]}\n\n{"query": "q2"}\n')
    items = load_dataset(path)
    assert len(items) == 2
    assert items[0].query == "q1"
    assert items[0].relevant_document_ids == ["d1"]


def test_load_dataset_rejects_invalid_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match="line 1"):
        load_dataset(path)


def test_load_dataset_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="No evaluation items"):
        load_dataset(path)


# --- runner -------------------------------------------------------------


class FakeService:
    def __init__(self, docs_by_query):
        self.docs_by_query = docs_by_query
        self.calls = []

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["query"]
        docs = self.docs_by_query.get(query, [])
        return SimpleNamespace(
            query=query,
            retrieved_documents=docs,
            reranked_documents=docs,
            context=SimpleNamespace(text="ctx", sources=[]),
            prompt="PROMPT",
        )


def make_items():
    return [
        EvalItem(query="q1", relevant_document_ids=["d1"]),
        EvalItem(query="q2", relevant_document_ids=["d2"]),
    ]


def test_evaluate_retrieval_runs_each_search_type():
    docs = {"q1": [make_doc("d1", "c1")], "q2": [make_doc("d2", "c2")]}
    service = FakeService(docs)
    evaluator = RAGEvaluator(service, k=3)

    results = evaluator.evaluate_retrieval(make_items(), ["dense", "hybrid"], rerank=False)

    assert len(results) == 4
    assert {r.search_type for r in results} == {"dense", "hybrid"}
    assert all(r.chunk_metrics is None for r in results)
    assert all(r.document_metrics["hit_rate"] == 1.0 for r in results)
    # each item was queried with the conversation scope and search type
    for call in service.calls:
        assert call["mode"] is not None
        assert call["limit"] == 3


def test_evaluate_quality_uses_generation_llm_and_judges(monkeypatch):
    docs = {"q1": [make_doc("d1", "c1")]}
    service = FakeService(docs)

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="generated answer")

    monkeypatch.setattr(
        "evaluation.runner.score_faithfulness", lambda *a, **k: SimpleNamespace(score=0.9)
    )
    monkeypatch.setattr(
        "evaluation.runner.score_answer_relevance", lambda *a, **k: SimpleNamespace(score=0.8)
    )
    monkeypatch.setattr(
        "evaluation.runner.score_context_relevance", lambda *a, **k: SimpleNamespace(score=0.7)
    )

    evaluator = RAGEvaluator(service, k=3, generation_llm=FakeLLM())
    quality = evaluator.evaluate_quality(
        [EvalItem(query="q1", relevant_document_ids=["d1"])],
        "hybrid",
        rerank=False,
        model=None,
        model_provider=None,
        api_key=None,
    )

    assert len(quality) == 1
    assert quality[0].answer == "generated answer"
    assert quality[0].faithfulness == 0.9
    assert quality[0].answer_relevance == 0.8
    assert quality[0].context_relevance == 0.7


# --- report --------------------------------------------------------------


def test_build_report_and_markdown():
    retrieval = [
        RetrievalResult(
            search_type="hybrid",
            rerank=False,
            query="q1",
            retrieved_count=1,
            chunk_metrics=None,
            document_metrics={
                "hit_rate": 1.0,
                "recall": 1.0,
                "precision": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
            },
        )
    ]
    report = build_report(
        config={
            "dataset": "x.jsonl",
            "k": 3,
            "search_types": ["hybrid"],
            "rerank": False,
            "judge": False,
        },
        retrieval=retrieval,
        quality=[],
    )
    assert "hybrid/no-rerank:document" in report["retrieval"]
    assert report["retrieval"]["hybrid/no-rerank:document"]["hit_rate"] == 1.0

    markdown = render_markdown(report)
    assert "# RAG Evaluation Report" in markdown
    assert "hybrid/no-rerank:document" in markdown
