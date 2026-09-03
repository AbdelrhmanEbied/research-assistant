import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import ClientDisconnect

import app.backend.services.chat_service as cs
from agent.llms import get_request_api_key
from app.backend.database.base import Base
from app.backend.database.models import Conversation
from app.backend.database.repositories import ConversationRepository, MessageRepository
from app.backend.schemas.chat import AgentMode, ChatRequest, LLMConfig, RegenerateRequest
from app.backend.services.chat_service import (
    DETAILS_MARKER,
    ERROR_MARKER,
    SOURCES_MARKER,
    THINKING_MARKER,
    ChatService,
)


@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda conn: conn.execute(
                Conversation.__table__.insert(), {"title": "existing"}
            )
        )
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db(session_factory):
    async with session_factory() as session:
        yield session


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
    async def astream_events(self, state, config=None, version="v2"):
        if False:  # pragma: no cover - makes this an async generator
            yield
        raise RuntimeError("boom")


class ThinkingFakeGraph(FakeGraph):
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


@pytest.mark.asyncio
async def test_stream_yields_answer_and_sources_marker(db, monkeypatch):
    sources = [
        {"source": "rag", "label": "a.pdf", "url": None, "document_id": "1"},
        {"source": "web", "label": "Some site", "url": "https://example.com", "document_id": None},
    ]
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph(sources)
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
        chunks.append(chunk)
    text = "".join(chunks)

    marker_idx = text.index(SOURCES_MARKER)
    assert text[:marker_idx].rstrip() == "answer here"

    details_idx = text.index(DETAILS_MARKER)
    payload = json.loads(text[marker_idx + len(SOURCES_MARKER) : details_idx].strip())
    assert payload == sources

    details = json.loads(text[details_idx + len(DETAILS_MARKER) :].strip())
    assert details["model"] == "fake"


@pytest.mark.asyncio
async def test_stream_passes_full_history_to_graph(db, session_factory, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    async with session_factory() as seed_db:
        repo = MessageRepository(seed_db)
        await repo.add_message(1, "user", "hi")
        await repo.add_message(1, "assistant", "hello!")

    graph = FakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    async for _ in service.stream(ChatRequest(query="how are you", conversation_id=1)):
        pass

    state = graph.received_state
    assert state["query"] == "how are you"
    assert state["conversation_id"] == "1"
    assert state["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]


@pytest.mark.asyncio
async def test_stream_keeps_api_key_out_of_graph_state(db, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    async for _ in service.stream(
        ChatRequest(
            query="q",
            conversation_id=1,
            llm_config=LLMConfig(model="m", model_provider="openai", api_key="secret"),
        )
    ):
        pass

    assert graph.received_state["llm_config"] == {
        "model": "m",
        "model_provider": "openai",
    }
    assert "api_key" not in graph.received_state["llm_config"]
    assert get_request_api_key() is None


async def _get_messages(session_factory, conversation_id):
    async with session_factory() as db:
        messages = await MessageRepository(db).list_for_history(conversation_id)
        return [
            {"id": m.id, "role": m.role, "content": m.content, "extra": m.extra}
            for m in messages
        ]


@pytest.mark.asyncio
async def test_regenerate_reuses_last_user_message_without_duplicating(
    db, session_factory, monkeypatch
):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    async with session_factory() as seed_db:
        repo = MessageRepository(seed_db)
        await repo.add_message(1, "user", "q1")
        await repo.add_message(1, "assistant", "a1")
        await repo.add_message(1, "user", "q2")
        await repo.add_message(1, "assistant", "old answer")

    graph = FakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    async for _ in service.regenerate(RegenerateRequest(conversation_id=1)):
        pass

    state = graph.received_state
    assert state["query"] == "q2"
    assert state["history"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    remaining = await _get_messages(session_factory, 1)
    roles = [m["role"] for m in remaining]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert remaining[-1]["content"] == "answer here"
    assert remaining[-2]["content"] == "q2"
    assert [m["content"] for m in remaining].count("q2") == 1


@pytest.mark.asyncio
async def test_regenerate_without_user_message_raises(db, session_factory, monkeypatch):
    async with session_factory() as seed_db:
        await MessageRepository(seed_db).add_message(1, "assistant", "only assistant")

    service = ChatService(graph=None, db=db, rag=None)

    async def _run():
        async for _ in service.regenerate(RegenerateRequest(conversation_id=1)):
            pass

    with pytest.raises(ValueError, match="No user message"):
        await _run()


@pytest.mark.asyncio
async def test_stream_on_client_disconnect_does_not_persist_partial_answer(
    db, session_factory, monkeypatch
):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = RaisingGraph(ClientDisconnect())
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    with pytest.raises(ClientDisconnect):
        async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
            chunks.append(chunk)
    assert "".join(chunks) == "partial"

    remaining = await _get_messages(session_factory, 1)
    assert [m["role"] for m in remaining] == ["user"]
    assert get_request_api_key() is None


@pytest.mark.asyncio
async def test_export_markdown_and_json(db, session_factory, monkeypatch):
    async with session_factory() as seed_db:
        repo = MessageRepository(seed_db)
        await repo.add_message(1, "user", "q1")
        await repo.add_message(1, "assistant", "a1")

    service = ChatService(graph=None, db=db, rag=None)

    md = await service.export_conversation(1, "markdown")
    js = await service.export_conversation(1, "json")

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


@pytest.mark.asyncio
async def test_stream_persists_sources_and_details_on_message(db, session_factory, monkeypatch):
    sources = [
        {"source": "rag", "label": "a.pdf", "url": None, "document_id": "1"},
    ]
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph(sources)
    service = ChatService(graph=graph, db=db, rag=None)

    async for _ in service.stream(ChatRequest(query="q", conversation_id=1)):
        pass

    remaining = await _get_messages(session_factory, 1)
    assistant = remaining[-1]
    assert assistant["content"] == "answer here"
    assert assistant["extra"]["sources"] == sources
    assert assistant["extra"]["details"]["model"] == "fake"


@pytest.mark.asyncio
async def test_stream_yields_error_marker_when_generation_fails(db, session_factory, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FailingGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
        chunks.append(chunk)
    text = "".join(chunks)

    err_idx = text.index(ERROR_MARKER)
    payload = json.loads(text[err_idx + len(ERROR_MARKER) :].strip())
    assert payload["message"] == "boom"

    remaining = await _get_messages(session_factory, 1)
    assert [m["role"] for m in remaining] == ["user"]


def test_generate_title_rejects_placeholder_output(monkeypatch):
    def _fake_llm(output):
        return SimpleNamespace(model="fake", invoke=lambda prompt: SimpleNamespace(content=output))

    service = ChatService(graph=None, db=None, rag=None)

    async def _run(output):
        monkeypatch.setattr(
            cs,
            "get_llms",
            lambda **kwargs: (_fake_llm(output), SimpleNamespace()),
        )
        return await service.generate_title("hi")

    assert asyncio.run(_run("New Chat")) == "hi"
    assert asyncio.run(_run("new chat")) == "hi"

    assert asyncio.run(_run("Greeting")) == "Greeting"


@pytest.mark.asyncio
async def test_stream_thinking_emits_marker_and_answer_only(db, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = ThinkingFakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    async for chunk in service.stream(
        ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
    ):
        chunks.append(chunk)

    text = "".join(chunks)

    marker_idx = text.rindex(THINKING_MARKER)
    assert text.count(THINKING_MARKER) == 3
    assert "Hmm, let" in text[:marker_idx]
    assert "me think" in text[:marker_idx]
    assert "Calling calculator..." in text[:marker_idx]
    assert "Hmm, let" not in text[marker_idx:]
    assert "answer here" in text[marker_idx:]
    assert "answer here" not in text[:marker_idx]

    assert graph.received_state["agent_mode"] == "thinking"


@pytest.mark.asyncio
async def test_stream_thinking_persists_thinking_and_details(db, session_factory, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = ThinkingFakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    async for _ in service.stream(
        ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
    ):
        pass

    remaining = await _get_messages(session_factory, 1)
    assistant = remaining[-1]
    assert assistant["content"] == "answer here"
    assert assistant["extra"]["thinking"] == "Hmm, let me think\n\nCalling calculator..."
    assert assistant["extra"]["details"]["agent_mode"] == "thinking"


@pytest.mark.asyncio
async def test_stream_thinking_without_thoughts_skips_thinking_section(db, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = NoThoughtsGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    async for chunk in service.stream(
        ChatRequest(query="q", conversation_id=1, agent_mode=AgentMode.THINKING)
    ):
        chunks.append(chunk)

    text = "".join(chunks)
    assert THINKING_MARKER not in text
    assert text[: text.index(DETAILS_MARKER)].rstrip() == "answer here"


@pytest.mark.asyncio
async def test_stream_fast_mode_never_emits_thinking_marker(db, monkeypatch):
    monkeypatch.setattr(
        cs,
        "get_llms",
        lambda **kwargs: (SimpleNamespace(model="fake"), SimpleNamespace()),
    )

    graph = FakeGraph([])
    service = ChatService(graph=graph, db=db, rag=None)

    chunks = []
    async for chunk in service.stream(ChatRequest(query="q", conversation_id=1)):
        chunks.append(chunk)

    text = "".join(chunks)
    assert THINKING_MARKER not in text
    assert text[: text.index(DETAILS_MARKER)].rstrip() == "answer here"
    assert graph.received_state["agent_mode"] is None
