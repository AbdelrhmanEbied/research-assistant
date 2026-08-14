from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.agent_schemas import KnowledgeSource, PromptMode, RouteQuery
from agent.graph import build_agent_graph
from rag.rag_schemas import Context, KnowledgeResult, RetrievedDocuments, build_sources
from settings import reset_settings_store


@pytest.fixture(autouse=True)
def _isolated_settings():
    reset_settings_store()
    yield
    reset_settings_store()


def make_result(query, docs, prompt="PROMPT"):
    return KnowledgeResult(
        query=query,
        retrieved_documents=docs,
        reranked_documents=docs,
        context=Context(text="ctx", sources=[]),
        prompt=prompt,
    )


def make_doc(text, metadata):
    return RetrievedDocuments(text=text, metadata=metadata, score=1.0)


class FakeRAG:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def prepare(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeSearchService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeGenLLM:
    def __init__(self, answer):
        self.answer = answer
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.answer, usage_metadata=None)


def make_graph(monkeypatch, *, source, docs, search_calls_expected, gen_answer="the answer"):
    rag_result = make_result("q", docs)
    rag = FakeRAG(rag_result)
    search = FakeSearchService(
        make_result("q", [make_doc("w", {"title": "T", "url": "https://x"})])
    )

    gen = FakeGenLLM(gen_answer)
    cls = SimpleNamespace(invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=source))
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())
    return graph, gen, rag, search


@pytest.mark.parametrize(
    "source,search_calls_expected",
    [
        (KnowledgeSource.RAG, False),
        (KnowledgeSource.WEB, True),
        (KnowledgeSource.NONE, False),
    ],
)
def test_graph_routes_by_classified_source(monkeypatch, source, search_calls_expected):
    docs = [make_doc("alpha", {"name": "a.pdf", "document_id": "1"})]
    graph, gen, rag, search = make_graph(
        monkeypatch, source=source, docs=docs, search_calls_expected=search_calls_expected
    )

    result = graph.invoke(
        {"query": "hello", "conversation_id": "1", "history": []},
        config={"configurable": {"thread_id": "1"}},
    )

    assert result["response"] == "the answer"
    if search_calls_expected:
        assert len(search.calls) == 1
        assert rag.calls == []
    else:
        assert len(rag.calls) == 1
        assert search.calls == []
    assert gen.last_prompt == "PROMPT"


def test_graph_passes_balanced_history_into_prompt(monkeypatch):
    docs = []
    graph, gen, rag, search = make_graph(
        monkeypatch,
        source=KnowledgeSource.NONE,
        docs=docs,
        search_calls_expected=False,
    )
    history = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first assistant"},
        {"role": "user", "content": "second user"},
    ]

    result = graph.invoke(
        {
            "query": "second user",
            "conversation_id": "1",
            "history": history,
            "llm_config": None,
        },
        config={"configurable": {"thread_id": "1"}},
    )

    assert result["response"] == "the answer"
    # the rag.prepare call must have received the full balanced history
    prepare_kwargs = rag.calls[0]
    assert prepare_kwargs["history"] == history
    assert prepare_kwargs["retrieve"] is False


def test_graph_exposes_sources_in_final_state(monkeypatch):
    docs = [
        make_doc("alpha", {"name": "a.pdf", "document_id": "1"}),
        make_doc("web", {"title": "Some site", "url": "https://example.com"}),
    ]
    graph, gen, rag, search = make_graph(
        monkeypatch,
        source=KnowledgeSource.RAG,
        docs=docs,
        search_calls_expected=False,
    )

    result = graph.invoke(
        {"query": "q", "conversation_id": "1", "history": []},
        config={"configurable": {"thread_id": "1"}},
    )

    expected = build_sources(make_result("q", docs))
    assert result["sources"] == expected
    labels = [s["label"] for s in result["sources"]]
    assert "a.pdf" in labels
    assert "Some site" in labels


def test_graph_replaces_history_across_invocations(monkeypatch):
    graph, gen, rag, search = make_graph(
        monkeypatch,
        source=KnowledgeSource.NONE,
        docs=[],
        search_calls_expected=False,
    )
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke(
        {
            "query": "first",
            "conversation_id": "1",
            "history": [{"role": "user", "content": "first"}],
        },
        config=config,
    )
    graph.invoke(
        {
            "query": "second",
            "conversation_id": "1",
            "history": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "old reply"},
                {"role": "user", "content": "second"},
            ],
        },
        config=config,
    )

    prepare_kwargs = rag.calls[-1]
    assert prepare_kwargs["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "second"},
    ]


def test_graph_respects_source_override_without_classifier(monkeypatch):
    docs = [make_doc("alpha", {"name": "a.pdf", "document_id": "1"})]
    rag_result = make_result("q", docs)
    rag = FakeRAG(rag_result)
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")

    def _assert_not_called(*_args, **_kwargs):
        raise AssertionError("classifier should not be called when both overrides are set")

    cls = SimpleNamespace(invoke=_assert_not_called)
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    result = graph.invoke(
        {
            "query": "compare these",
            "conversation_id": "1",
            "history": [],
            "mode_override": "compare",
            "source_override": "documents",
        },
        config={"configurable": {"thread_id": "1"}},
    )

    assert result["response"] == "the answer"
    prepare_kwargs = rag.calls[0]
    assert prepare_kwargs["retrieve"] is True
    assert prepare_kwargs["mode"] == PromptMode.COMPARE


def test_graph_respects_chat_override_skips_retrieval(monkeypatch):
    rag = FakeRAG(make_result("q", []))
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("hi")

    def _assert_not_called(*_args, **_kwargs):
        raise AssertionError("classifier should not run for a forced chat")

    cls = SimpleNamespace(invoke=_assert_not_called)
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {
            "query": "just chat",
            "conversation_id": "1",
            "history": [],
            "source_override": "chat",
        },
        config={"configurable": {"thread_id": "1"}},
    )

    assert search.calls == []
    # NONE still flows through rag.prepare but without retrieval
    assert len(rag.calls) == 1
    assert rag.calls[0]["retrieve"] is False
    assert gen.last_prompt == "PROMPT"


def test_graph_passes_retrieval_config_to_rag(monkeypatch):
    docs = [make_doc("alpha", {"name": "a.pdf", "document_id": "1"})]
    rag_result = make_result("q", docs)
    rag = FakeRAG(rag_result)
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")
    cls = SimpleNamespace(
        invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=KnowledgeSource.RAG)
    )
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {
            "query": "q",
            "conversation_id": "1",
            "history": [],
            "retrieval_config": {"search_type": "sparse", "limit": 7, "rerank": False},
        },
        config={"configurable": {"thread_id": "1"}},
    )

    prepare_kwargs = rag.calls[0]
    assert prepare_kwargs["search_type"] == "sparse"
    assert prepare_kwargs["limit"] == 7
    assert prepare_kwargs["rerank"] is False


def test_graph_defaults_retrieval_config_from_settings(monkeypatch):
    docs = [make_doc("alpha", {"name": "a.pdf", "document_id": "1"})]
    rag = FakeRAG(make_result("q", docs))
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")
    cls = SimpleNamespace(
        invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=KnowledgeSource.RAG)
    )
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {"query": "q", "conversation_id": "1", "history": []},
        config={"configurable": {"thread_id": "1"}},
    )

    prepare_kwargs = rag.calls[0]
    assert prepare_kwargs["search_type"] == "hybrid"
    assert prepare_kwargs["rerank"] is True
    assert 1 <= prepare_kwargs["limit"] <= 50


def test_graph_passes_max_results_to_web_search(monkeypatch):
    rag = FakeRAG(make_result("q", []))
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")
    cls = SimpleNamespace(
        invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=KnowledgeSource.WEB)
    )
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {
            "query": "news",
            "conversation_id": "1",
            "history": [],
            "retrieval_config": {"limit": 4},
        },
        config={"configurable": {"thread_id": "1"}},
    )

    search_kwargs = search.calls[0]
    assert search_kwargs["max_results"] == 4


def test_graph_passes_search_depth_to_web_search(monkeypatch):
    rag = FakeRAG(make_result("q", []))
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")
    cls = SimpleNamespace(
        invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=KnowledgeSource.WEB)
    )
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {
            "query": "news",
            "conversation_id": "1",
            "history": [],
            "retrieval_config": {"search_depth": "advanced"},
        },
        config={"configurable": {"thread_id": "1"}},
    )

    search_kwargs = search.calls[0]
    assert search_kwargs["search_depth"] == "advanced"


def test_graph_defaults_search_depth_to_basic(monkeypatch):
    rag = FakeRAG(make_result("q", []))
    search = FakeSearchService(make_result("q", []))
    gen = FakeGenLLM("the answer")
    cls = SimpleNamespace(
        invoke=lambda msgs: RouteQuery(mode=PromptMode.CHAT, source=KnowledgeSource.WEB)
    )
    monkeypatch.setattr("agent.nodes.get_llms", lambda **kwargs: (gen, cls))

    graph = build_agent_graph(rag=rag, search_service=search, checkpointer=InMemorySaver())

    graph.invoke(
        {"query": "news", "conversation_id": "1", "history": []},
        config={"configurable": {"thread_id": "1"}},
    )

    search_kwargs = search.calls[0]
    assert search_kwargs["search_depth"] == "basic"
