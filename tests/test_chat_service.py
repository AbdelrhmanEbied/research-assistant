import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import ClientDisconnect

import app.backend.services.chat_service as cs
from agent.llms import get_request_api_key
from app.backend.database.base import Base
from app.backend.database.models import Conversation
from app.backend.database.repositories import MessageRepository
from app.backend.schemas.chat import AgentMode, ChatRequest, LLMConfig, RegenerateRequest
from app.backend.services.chat_service import (
    DETAILS_MARKER,
    ERROR_MARKER,
    SOURCES_MARKER,
    THINKING_MARKER,
    ChatService,
)


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        db.add(Conversation(title="existing"))
        db.commit()
    return factory


class FakeGraph:
    def __init__(self, sources):
        self.sources = sources
        self.received_state = None

    async def astream_events(self, state, config=None, version="v2"):
        self.received_state = state
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "answer "}])},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "here"}])},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "prepare_prompt"},
            "data": {"output": {"sources": self.sources}},
        }


class RaisingGraph(FakeGraph):
    """Graph that streams one token then disconnects mid-stream."""

    def __init__(self, error):
        super().__init__([])
        self.error = error

    async def astream_events(self, state, config=None, version="v2"):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate_answer"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "partial"}])},
        }
        raise self.error


class FailingGraph(FakeGraph):
    """Graph that fails before streaming any content."""

    async def astream_events(self, state, config=None, version="v2"):
        if False:  # pragma: no cover - makes this an async generator
            yield
        raise RuntimeError("boom")


class ThinkingFakeGraph(FakeGraph):
    """Streams a Gemini-style thinking turn (tool call), tool status, then the final answer.

    Mirrors the real Gemini 3.5 stream: the reasoning turn carries thought
    content as ``{"type": "thinking"}`` blocks and ends in a tool call; the
    final answer turn is plain text blocks.
    """

    async def astream_events(self, state, config=None, version="v2"):
        self.received_state = state
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {
                "chunk": AIMessageChunk(content=[{"type": "thinking", "thinking": "Hmm, let"}])
            },
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {
                "chunk": AIMessageChunk(content=[{"type": "thinking", "thinking": " me think"}])
            },
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {
                "output": AIMessage(
                    content=[{"type": "thinking", "thinking": "Hmm, let me think"}],
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "2 + 2"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            },
        }
        yield {
            "event": "on_tool_start",
            "name": "calculator",
            "metadata": {"langgraph_node": "execute_tools"},
            "data": {"input": {"expression": "2 + 2"}},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "answer "}])},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "here"}])},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"output": AIMessage(content=[{"type": "text", "text": "answer here"}])},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "prepare_prompt"},
            "data": {"output": {"sources": self.sources}},
        }


class NoThoughtsGraph(FakeGraph):
    """A model that never exposes thinking blocks (e.g. non-Gemini provider)."""

    async def astream_events(self, state, config=None, version="v2"):
        self.received_state = state
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "answer "}])},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text", "text": "here"}])},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent_reason"},
            "data": {"output": AIMessage(content=[{"type": "text", "text": "answer here"}])},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "prepare_prompt"},
            "data": {"output": {"sources": self.sources}},
        }


def test_stream_yields_answer_and_sources_marker(session_factory, monkeypatch):
    sources = [
        {"source": "rag", "label": "a.pdf", "url": None, "document_id": "1"},
        {"source": "web", "label": "Some site", "url": "https://example.com", "document_id": None},
    ]
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph(sources)
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    text = "".join(chunks)

    marker_idx = text.index(SOURCES_MARKER)
    assert text[:marker_idx].rstrip() == "answer here"

    details_idx = text.index(DETAILS_MARKER)
    payload = json.loads(text[marker_idx + len(SOURCES_MARKER) : details_idx].strip())
    assert payload == sources

    details = json.loads(text[details_idx + len(DETAILS_MARKER) :].strip())
    assert details["model"] == "fake"


def test_stream_passes_full_history_to_graph(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _seed():
        def _do():
            db = session_factory()
            try:
                from app.backend.database.repositories import MessageRepository

                MessageRepository(db).add_message(1, "user", "hi")
                MessageRepository(db).add_message(1, "assistant", "hello!")
            finally:
                db.close()

        return await cs.ChatService._run_db(_do)

    asyncio.run(_seed())

    async def _run():
        async for _ in service.stream(ChatRequest(query="how are you", conversation_id=1)):
            pass

    asyncio.run(_run())

    state = graph.received_state
    assert state["query"] == "how are you"
    assert state["conversation_id"] == "1"
    # prior turns (excluding the just-persisted user message) come through
    assert state["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]


def test_stream_keeps_api_key_out_of_graph_state(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        async for _ in service.stream(
            ChatRequest(
                query="q",
                conversation_id=1,
                llm_config=LLMConfig(model="m", model_provider="openai", api_key="secret"),
            )
        ):
            pass

    asyncio.run(_run())

    assert graph.received_state["llm_config"] == {
        "model": "m",
        "model_provider": "openai",
    }
    assert "api_key" not in graph.received_state["llm_config"]
    # the key never leaks into the persisted checkpoint state
    assert get_request_api_key() is None


def _messages(session_factory, conversation_id):
    def _do():
        db = session_factory()
        try:
            return [
                {"id": m.id, "role": m.role, "content": m.content, "extra": m.extra}
                for m in MessageRepository(db).list_for_history(conversation_id)
            ]
        finally:
            db.close()

    return asyncio.run(cs.ChatService._run_db(_do))


def test_regenerate_reuses_last_user_message_without_duplicating(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    def _seed():
        def _do():
            db = session_factory()
            try:
                repo = MessageRepository(db)
                repo.add_message(1, "user", "q1")
                repo.add_message(1, "assistant", "a1")
                repo.add_message(1, "user", "q2")
                repo.add_message(1, "assistant", "old answer")
            finally:
                db.close()

        return asyncio.run(cs.ChatService._run_db(_do))

    _seed()

    graph = FakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        async for _ in service.regenerate(RegenerateRequest(conversation_id=1)):
            pass

    asyncio.run(_run())

    state = graph.received_state
    assert state["query"] == "q2"
    assert state["history"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    remaining = _messages(session_factory, 1)
    roles = [m["role"] for m in remaining]
    # old assistant answer dropped, no duplicate user message
    assert roles == ["user", "assistant", "user", "assistant"]
    assert remaining[-1]["content"] == "answer here"
    assert remaining[-2]["content"] == "q2"
    assert [m["content"] for m in remaining].count("q2") == 1


def test_regenerate_without_user_message_raises(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)

    def _seed():
        def _do():
            db = session_factory()
            try:
                MessageRepository(db).add_message(1, "assistant", "only assistant")
            finally:
                db.close()

        return asyncio.run(cs.ChatService._run_db(_do))

    _seed()

    service = ChatService(graph=None, rag=None)

    async def _run():
        async for _ in service.regenerate(RegenerateRequest(conversation_id=1)):
            pass

    with pytest.raises(ValueError, match="No user message"):
        asyncio.run(_run())


def test_stream_on_client_disconnect_does_not_persist_partial_answer(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = RaisingGraph(ClientDisconnect())
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        with pytest.raises(ClientDisconnect):
            async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
                chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    assert "".join(chunks) == "partial"

    remaining = _messages(session_factory, 1)
    # the user message was persisted, but no partial assistant answer
    assert [m["role"] for m in remaining] == ["user"]
    assert get_request_api_key() is None


def test_export_markdown_and_json(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)

    def _seed():
        def _do():
            db = session_factory()
            try:
                repo = MessageRepository(db)
                repo.add_message(1, "user", "q1")
                repo.add_message(1, "assistant", "a1")
            finally:
                db.close()

        return asyncio.run(cs.ChatService._run_db(_do))

    _seed()

    service = ChatService(graph=None, rag=None)

    async def _run():
        md = await service.export_conversation(1, "markdown")
        js = await service.export_conversation(1, "json")
        return md, js

    md, js = asyncio.run(_run())

    assert "# " in md
    assert "## User" in md
    assert "q1" in md
    assert "## Assistant" in md
    assert "a1" in md

    data = json.loads(js)
    assert data["title"] == "existing"
    assert data["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_stream_persists_sources_and_details_on_message(session_factory, monkeypatch):
    sources = [
        {"source": "rag", "label": "a.pdf", "url": None, "document_id": "1"},
    ]
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph(sources)
    service = ChatService(graph=graph, rag=None)

    async def _run():
        async for _ in service.stream(ChatRequest(query="q", conversation_id=1)):
            pass

    asyncio.run(_run())

    remaining = _messages(session_factory, 1)
    assistant = remaining[-1]
    assert assistant["content"] == "answer here"
    assert assistant["extra"]["sources"] == sources
    assert assistant["extra"]["details"]["model"] == "fake"


def test_stream_yields_error_marker_when_generation_fails(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FailingGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    text = "".join(chunks)

    # the error is surfaced as a marker instead of killing the stream
    err_idx = text.index(ERROR_MARKER)
    payload = json.loads(text[err_idx + len(ERROR_MARKER) :].strip())
    assert payload["message"] == "boom"

    # no assistant answer is persisted, only the user message
    remaining = _messages(session_factory, 1)
    assert [m["role"] for m in remaining] == ["user"]


def test_generate_title_rejects_placeholder_output(monkeypatch):
    """A model echoing 'New Chat' must not become the stored title."""

    def _fake_llm(output):
        return SimpleNamespace(model="fake", invoke=lambda prompt: SimpleNamespace(content=output))

    service = ChatService(graph=None, rag=None)

    async def _run(output):
        monkeypatch.setattr(
            cs,
            "get_llms",
            lambda **kwargs: (_fake_llm(output), SimpleNamespace()),
        )
        return await service.generate_title("hi")

    # the placeholder echo is discarded in favour of the query fallback
    assert asyncio.run(_run("New Chat")) == "hi"
    assert asyncio.run(_run("new chat")) == "hi"

    # a real title is kept
    assert asyncio.run(_run("Greeting")) == "Greeting"


def test_stream_thinking_emits_marker_and_answer_only(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = ThinkingFakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        async for chunk in service.stream(
            ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
        ):
            chunks.append(chunk)
        return chunks

    text = "".join(asyncio.run(_run()))

    assert text.count(THINKING_MARKER) == 3  # 2 thought chunks + 1 tool status
    # thinking text precedes the last marker; the answer follows it
    marker_idx = text.rindex(THINKING_MARKER)
    assert "Hmm, let" in text[:marker_idx]
    assert "me think" in text[:marker_idx]
    assert "Calling calculator..." in text[:marker_idx]
    assert "Hmm, let" not in text[marker_idx:]
    assert "answer here" in text[marker_idx:]
    assert "answer here" not in text[:marker_idx]

    # agent_mode reached the graph state
    assert graph.received_state["agent_mode"] == "thinking"


def test_stream_thinking_persists_thinking_and_details(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = ThinkingFakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        async for _ in service.stream(
            ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
        ):
            pass

    asyncio.run(_run())

    remaining = _messages(session_factory, 1)
    assistant = remaining[-1]
    assert assistant["content"] == "answer here"
    assert assistant["extra"]["thinking"] == "Hmm, let me think\n\nCalling calculator..."
    assert assistant["extra"]["details"]["agent_mode"] == "thinking"


def test_stream_thinking_without_thoughts_skips_thinking_section(session_factory, monkeypatch):
    """A model that exposes no thought blocks streams the answer with no markers."""
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = NoThoughtsGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        async for chunk in service.stream(
            ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
        ):
            chunks.append(chunk)
        return chunks

    text = "".join(asyncio.run(_run()))
    assert THINKING_MARKER not in text
    assert text[: text.index(DETAILS_MARKER)].rstrip() == "answer here"


def test_stream_fast_mode_never_emits_thinking_marker(session_factory, monkeypatch):
    monkeypatch.setattr(cs, "SessionLocal", session_factory)
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph([])
    service = ChatService(graph=graph, rag=None)

    async def _run():
        chunks = []
        async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
            chunks.append(chunk)
        return chunks

    text = "".join(asyncio.run(_run()))
    assert THINKING_MARKER not in text
    assert text[: text.index(DETAILS_MARKER)].rstrip() == "answer here"
    assert graph.received_state["agent_mode"] is None
