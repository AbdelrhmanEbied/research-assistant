from types import SimpleNamespace

from agent.agent_schemas import PromptMode
from rag.rag_schemas import Context, RetrievedDocuments, SearchType
from rag.rag_service import RAGService
from rag.reranker import Reranker
from rag.retriever import Retriever


def point(document_id, text, score, chunk_index=0):
    return SimpleNamespace(
        payload={
            "text": text,
            "document_id": document_id,
            "name": f"{document_id}.pdf",
            "chunk_index": chunk_index,
        },
        score=score,
    )


class FakeVectorstore:
    """Replays in-memory points and records every search/list call."""

    def __init__(self, points_by_document, all_points):
        self.points_by_document = points_by_document
        self.all_points = all_points
        self.search_calls = []
        self.list_calls = []

    def list_document_ids(self, qdrant_filter=None):
        self.list_calls.append(qdrant_filter)
        return [(doc_id, f"{doc_id}.pdf") for doc_id in self.points_by_document]

    def search(self, query, search_type, limit, qdrant_filter=None):
        self.search_calls.append((query, search_type, limit, qdrant_filter))

        conditions = (qdrant_filter.must or []) if qdrant_filter else []
        document_id = None
        for condition in conditions:
            if condition.key == "document_id":
                document_id = condition.match.value

        pool = self.points_by_document[document_id] if document_id is not None else self.all_points

        ordered = sorted(pool, key=lambda p: p.score, reverse=True)
        return SimpleNamespace(points=ordered[:limit])


def make_retriever(store):
    return Retriever(embedder=SimpleNamespace(embed_query=lambda **kw: object()), vectorstore=store)


def make_doc(document_id, text, score):
    return RetrievedDocuments(
        text=text,
        metadata={"document_id": document_id, "name": f"{document_id}.pdf"},
        score=score,
    )


def test_grouped_retrieval_queries_each_document_independently():
    doc_a = [point("a", "a-top", 0.95), point("a", "a-second", 0.85)]
    doc_b = [point("b", "b-top", 0.80), point("b", "b-second", 0.70)]
    store = FakeVectorstore({"a": doc_a, "b": doc_b}, doc_a + doc_b)

    retriever = make_retriever(store)
    docs = retriever.retrieve(
        query="q",
        limit=2,
        search_type=SearchType.HYBRID,
        conversation_id="7",
        group_by_document=True,
    )

    # ceil(2 / 2) = 1 per document, merge-sorted by score
    assert len(docs) == 2
    assert [d.text for d in docs] == ["a-top", "b-top"]
    assert {d.metadata["document_id"] for d in docs} == {"a", "b"}

    # one query per document, each combining conversation + document scoping
    assert len(store.search_calls) == 2
    for _, _, _, qdrant_filter in store.search_calls:
        keys = {condition.key for condition in qdrant_filter.must}
        assert keys == {"conversation_id", "document_id"}
        assert qdrant_filter.must[0].match.value == "7"


def test_grouped_retrieval_preserves_conversation_filter_in_scope():
    doc_a = [point("a", "a1", 0.9)]
    doc_b = [point("b", "b1", 0.8)]
    store = FakeVectorstore({"a": doc_a, "b": doc_b}, doc_a + doc_b)

    retriever = make_retriever(store)
    retriever.retrieve(query="q", limit=2, conversation_id="42", group_by_document=True)

    scope_filter = store.list_calls[0]
    assert scope_filter is not None
    assert scope_filter.must[0].key == "conversation_id"
    assert scope_filter.must[0].match.value == "42"


def test_grouped_retrieval_merges_more_than_two_documents():
    doc_a = [point("a", "a1", 0.99), point("a", "a2", 0.1)]
    doc_b = [point("b", "b1", 0.5)]
    doc_c = [point("c", "c1", 0.4)]
    store = FakeVectorstore({"a": doc_a, "b": doc_b, "c": doc_c}, doc_a + doc_b + doc_c)

    retriever = make_retriever(store)
    docs = retriever.retrieve(query="q", limit=3, group_by_document=True)

    # ceil(3 / 3) = 1 per doc, so each document contributes
    assert len(docs) == 3
    assert {d.metadata["document_id"] for d in docs} == {"a", "b", "c"}


def test_grouped_retrieval_single_document_uses_normal_path():
    doc_a = [point("a", "a1", 0.9), point("a", "a2", 0.8)]
    store = FakeVectorstore({"a": doc_a}, doc_a)

    retriever = make_retriever(store)
    docs = retriever.retrieve(query="q", limit=3, group_by_document=True)

    # a single in-scope document falls back to one unconstrained query
    assert len(store.search_calls) == 1
    _, _, limit, qdrant_filter = store.search_calls[0]
    assert limit == 3
    conditions = (qdrant_filter.must or []) if qdrant_filter else []
    assert all(condition.key != "document_id" for condition in conditions)
    assert len(docs) == 2


def test_retrieve_without_grouping_stays_single_query():
    doc_a = [point("a", "a1", 0.9)]
    doc_b = [point("b", "b1", 0.8)]
    store = FakeVectorstore({"a": doc_a, "b": doc_b}, doc_a + doc_b)

    retriever = make_retriever(store)
    retriever.retrieve(query="q", limit=2, conversation_id="1", group_by_document=False)

    assert len(store.search_calls) == 1
    assert store.list_calls == []


def test_rerank_diversified_guarantees_one_chunk_per_document():
    reranker = Reranker.__new__(Reranker)
    documents = [
        make_doc("a", "a-top", 0.95),
        make_doc("a", "a-mid", 0.70),
        make_doc("b", "b-top", 0.85),
        make_doc("c", "c-top", 0.60),
    ]
    reranker._score = lambda q, ds: list(zip(ds, [d.score for d in ds], strict=True))

    result = reranker.rerank_diversified("q", documents, top_k=3)

    assert len(result) == 3
    assert {d.metadata["document_id"] for d in result} == {"a", "b", "c"}
    # the top chunk of each document is chosen first
    assert [d.text for d in result] == ["a-top", "b-top", "c-top"]


def test_rerank_diversified_fills_by_relevance_after_guarantees():
    reranker = Reranker.__new__(Reranker)
    documents = [
        make_doc("a", "a-top", 0.95),
        make_doc("a", "a-second", 0.85),
        make_doc("b", "b-top", 0.80),
    ]
    reranker._score = lambda q, ds: list(zip(ds, [d.score for d in ds], strict=True))

    result = reranker.rerank_diversified("q", documents, top_k=3)

    assert [d.text for d in result] == ["a-top", "b-top", "a-second"]


def test_rerank_diversified_single_document_matches_plain_rerank():
    reranker = Reranker.__new__(Reranker)
    documents = [
        make_doc("a", "a1", 0.9),
        make_doc("a", "a2", 0.8),
        make_doc("a", "a3", 0.7),
    ]
    reranker._score = lambda q, ds: list(zip(ds, [d.score for d in ds], strict=True))

    diversified = reranker.rerank_diversified("q", documents, top_k=2)
    plain = reranker.rerank("q", documents, top_k=2)

    assert [d.text for d in diversified] == [d.text for d in plain] == ["a1", "a2"]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return []


class FakeReranker:
    def __init__(self):
        self.rerank_calls = []
        self.diversified_calls = []

    def rerank(self, **kwargs):
        self.rerank_calls.append(kwargs)
        return kwargs["documents"]

    def rerank_diversified(self, **kwargs):
        self.diversified_calls.append(kwargs)
        return kwargs["documents"]


def make_rag_service():
    retriever = FakeRetriever()
    reranker = FakeReranker()
    service = RAGService(
        loader=None,
        chunker=None,
        embedder=None,
        qdrant_manager=None,
        retriever=retriever,
        reranker=reranker,
        context_builder=SimpleNamespace(build=lambda **kw: Context(text="ctx", sources=[])),
        prompt_builder=SimpleNamespace(build=lambda **kw: "PROMPT"),
    )
    return service, retriever, reranker


def test_prepare_compare_mode_groups_and_diversifies():
    service, retriever, reranker = make_rag_service()

    service.prepare(
        query="compare docs",
        mode=PromptMode.COMPARE,
        history=[],
        conversation_id="3",
    )

    retrieve_kwargs = retriever.calls[0]
    assert retrieve_kwargs["group_by_document"] is True
    assert retrieve_kwargs["conversation_id"] == "3"
    assert len(reranker.diversified_calls) == 1
    assert reranker.rerank_calls == []


def test_prepare_other_modes_use_plain_retrieval_and_rerank():
    service, retriever, reranker = make_rag_service()

    service.prepare(query="sum up", mode=PromptMode.SUMMARIZE, history=[])

    assert retriever.calls[0]["group_by_document"] is False
    assert len(reranker.rerank_calls) == 1
    assert reranker.diversified_calls == []
