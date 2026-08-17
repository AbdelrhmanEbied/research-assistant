import uuid

from langchain_core.documents import Document

from agent.agent_schemas import PromptMode
from rag.builders import ContextBuilder, PromptBuilder
from rag.chunker import DocumentChunker
from rag.rag_schemas import Context, KnowledgeResult, RetrievedDocuments, build_sources


def make_doc(
    text: str,
    metadata: dict | None = None,
    score: float = 1.0,
) -> RetrievedDocuments:
    return RetrievedDocuments(text=text, metadata=metadata or {}, score=score)


def test_context_builder_renders_documents_and_sources():
    docs = [
        make_doc(
            "alpha content",
            {"name": "report.pdf", "page": 3, "document_id": "1", "chunk_id": "c1"},
        ),
        make_doc("beta content", {"name": "notes.txt", "page": 1, "source": "web"}),
    ]

    context = ContextBuilder().build(docs)

    assert "[Document 1]" in context.text
    assert "[Document 2]" in context.text
    assert "alpha content" in context.text
    assert "beta content" in context.text

    # internal metadata keys must not leak into the prompt context
    assert "document_id" not in context.text
    assert "chunk_id" not in context.text
    assert "conversation_id" not in context.text

    # useful metadata survives
    assert "Name" in context.text
    assert "Page" in context.text

    assert context.sources == [
        {"source": None, "page": 3, "chunk_id": "c1"},
        {"source": "web", "page": 1, "chunk_id": None},
    ]


def test_context_builder_skips_internal_keys():
    docs = [
        make_doc(
            "content",
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "created_at": "2026-01-01",
                "conversation_id": "7",
                "chunk_index": 0,
                "total_chunks": 5,
                "source": "/tmp/x.pdf",
                "name": "x.pdf",
            },
        )
    ]

    context = ContextBuilder().build(docs)

    assert "x.pdf" in context.text
    for key in (
        "document_id",
        "chunk_id",
        "created_at",
        "conversation_id",
        "chunk_index",
        "total_chunks",
    ):
        assert key not in context.text


def test_prompt_builder_supports_all_modes():
    builder = PromptBuilder(system_prompt="SYS")
    question = "What is RAG?"
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    context = ContextBuilder().build([make_doc("context text", {"name": "a.pdf"})])

    for mode in PromptMode:
        prompt = builder.build(mode=mode, question=question, context=context, history=history)
        assert "SYS" in prompt
        assert question in prompt
        assert "context text" in prompt
        assert "hello" in prompt
        assert "hi" in prompt
        assert prompt.rstrip().endswith("Assistant:")


def test_prompt_builder_rejects_unknown_mode():
    builder = PromptBuilder(system_prompt="SYS")
    try:
        builder.build(mode="bogus", question="q", context=None, history=[])
    except ValueError as exc:
        assert "Unsupported prompt mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_sources_includes_rag_and_web_metadata():
    rag_doc = make_doc(
        "alpha content " * 40,
        {
            "name": "report.pdf",
            "page": 3,
            "document_id": "7",
            "chunk_id": "c-7-2",
        },
    )
    web_doc = make_doc(
        "web snippet text",
        {"title": "Some Site", "url": "https://www.example.com/path"},
    )
    result = KnowledgeResult(
        query="q",
        retrieved_documents=[rag_doc, web_doc],
        reranked_documents=[rag_doc, web_doc],
        context=Context(text="", sources=[]),
        prompt="P",
    )

    sources = build_sources(result)

    rag_source = sources[0]
    assert rag_source["source"] == "rag"
    assert rag_source["document_id"] == "7"
    assert rag_source["chunk_id"] == "c-7-2"
    assert rag_source["page"] == 3
    assert rag_source["url"] is None
    assert rag_source["snippet"] and rag_source["snippet"].endswith("…")

    web_source = sources[1]
    assert web_source["source"] == "web"
    assert web_source["title"] == "Some Site"
    assert web_source["url"] == "https://www.example.com/path"
    assert web_source["domain"] == "example.com"
    assert web_source["snippet"] == "web snippet text"
    # no internal search objects leak into the citation
    assert "score" not in rag_source
    assert "raw_content" not in web_source


def test_chunker_assigns_deterministic_chunk_ids_per_document():
    text = "This is a document about hybrid retrieval systems. " * 60
    documents = [
        Document(page_content=text, metadata={"document_id": "doc-a", "name": "a.txt"}),
        Document(page_content=text, metadata={"document_id": "doc-b", "name": "b.txt"}),
    ]

    chunks = DocumentChunker(chunk_size=500, chunk_overlap=0).chunk(documents)

    assert len(chunks) > 2
    chunk_ids_a = [c.metadata["chunk_id"] for c in chunks if c.metadata["document_id"] == "doc-a"]
    chunk_ids_b = [c.metadata["chunk_id"] for c in chunks if c.metadata["document_id"] == "doc-b"]
    expected_a = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc-a:{i}")) for i in range(len(chunk_ids_a))
    ]
    expected_b = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc-b:{i}")) for i in range(len(chunk_ids_b))
    ]
    assert chunk_ids_a == expected_a
    assert chunk_ids_b == expected_b
