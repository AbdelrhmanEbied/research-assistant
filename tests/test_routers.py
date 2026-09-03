from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.backend.database.base import Base
from app.backend.database.database import get_db
from app.backend.database.models import Conversation, ConversationDocument, Document, Message
from app.backend.routers.chat_router import get_chat_service
from app.backend.routers.chat_router import router as chat_router
from app.backend.routers.document_router import get_document_service
from app.backend.routers.document_router import router as document_router


@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db(session_factory):
    async with session_factory() as session:
        yield session


@pytest.fixture
def chat_app(db):
    app = FastAPI()
    app.include_router(chat_router)

    async def override_get_db():
        yield db

    class FakeChatService:
        async def stream(self, request):
            yield "hello"
            yield " world"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    return app


@pytest.mark.asyncio
async def test_stream_chat(chat_app):
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/",
            json={"query": "hi", "conversation_id": 1},
        )
    assert response.status_code == 200
    assert response.text == "hello world"


@pytest.mark.asyncio
async def test_conversation_crud_and_message_pagination(db, session_factory):
    app = FastAPI()
    app.include_router(chat_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            return None

    app.state.checkpointer = FakeCheckpointer()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = (await client.post("/chat/conversations")).json()
        conversation_id = created["id"]

        assert (await client.get("/chat/list")).json()[0]["id"] == conversation_id

        base = datetime.now(UTC)
        async with session_factory() as seed_db:
            for i in range(5):
                seed_db.add(
                    Message(
                        conversation_id=conversation_id,
                        role="user" if i % 2 == 0 else "assistant",
                        content=f"msg-{i}",
                        created_at=base + timedelta(seconds=i),
                    )
                )
            await seed_db.commit()

        page = (await client.get(f"/chat/{conversation_id}/messages?limit=2&offset=0")).json()
        assert page["total"] == 5
        assert page["limit"] == 2
        assert page["offset"] == 0
        assert [m["content"] for m in page["messages"]] == ["msg-4", "msg-3"]

        page2 = (await client.get(f"/chat/{conversation_id}/messages?limit=2&offset=2")).json()
        assert [m["content"] for m in page2["messages"]] == ["msg-2", "msg-1"]

        assert (await client.delete(f"/chat/{conversation_id}")).status_code == 200
        assert (await client.delete(f"/chat/{conversation_id}")).status_code == 404


@pytest.mark.asyncio
async def test_conversation_delete_uses_checkpointer(db):
    app = FastAPI()
    app.include_router(chat_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    deleted_threads = []

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            deleted_threads.append(thread_id)

    app.state.checkpointer = FakeCheckpointer()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        conversation_id = (await client.post("/chat/conversations")).json()["id"]
        assert (await client.delete(f"/chat/{conversation_id}")).status_code == 200
        assert deleted_threads == [str(conversation_id)]


@pytest.fixture
def document_app(db):
    app = FastAPI()
    app.include_router(document_router)

    async def override_get_db():
        yield db

    class FakeDocumentService:
        def __init__(self):
            self.uploaded = []
            self.deleted = []

        async def upload_document(self, conversation_id, file):
            self.uploaded.append((conversation_id, file.filename))
            return {"id": 1, "name": file.filename}

        async def list_all_documents(self):
            return [
                {
                    "id": 1,
                    "name": "a.pdf",
                    "file_path": "/tmp/a.pdf",
                    "conversations": [],
                }
            ]

        async def delete_document(self, document_id):
            self.deleted.append(document_id)

    fake = FakeDocumentService()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_document_service] = lambda: fake
    return app, fake


@pytest.mark.asyncio
async def test_document_upload_and_delete(document_app):
    app, fake = document_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents/upload",
            data={"conversation_id": 1},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "notes.txt"
        assert fake.uploaded == [(1, "notes.txt")]

        assert (await client.get("/documents/")).json()[0]["name"] == "a.pdf"

        assert (await client.delete("/documents/1")).status_code == 200
        assert fake.deleted == [1]


@pytest.mark.asyncio
async def test_document_list_by_conversation(db, session_factory):
    app = FastAPI()
    app.include_router(document_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    async with session_factory() as seed_db:
        conversation = Conversation(title="conv")
        seed_db.add(conversation)
        await seed_db.commit()

        doc = Document(name="a.pdf", file_path="/tmp/a.pdf")
        seed_db.add(doc)
        await seed_db.commit()

        seed_db.add(ConversationDocument(conversation_id=conversation.id, document_id=doc.id))
        await seed_db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/documents/{conversation.id}/documents")
        assert response.status_code == 200
        assert response.json()[0]["name"] == "a.pdf"


@pytest.mark.asyncio
async def test_conversation_search_by_title_and_content(db, session_factory):
    app = FastAPI()
    app.include_router(chat_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    async with session_factory() as seed_db:
        conv_a = Conversation(title="Quantum computing notes")
        conv_b = Conversation(title="Cooking")
        seed_db.add_all([conv_a, conv_b])
        await seed_db.commit()

        seed_db.add(Message(conversation_id=conv_a.id, role="user", content="Tell me about qubits"))
        seed_db.add(Message(conversation_id=conv_b.id, role="user", content="How do I make pasta?"))
        await seed_db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        titles = [c["title"] for c in (await client.get("/chat/list?q=quantum")).json()]
        assert titles == ["Quantum computing notes"]

        titles = [c["title"] for c in (await client.get("/chat/list?q=qubits")).json()]
        assert titles == ["Quantum computing notes"]

        assert (await client.get("/chat/list?q=zzzzz")).json() == []


@pytest.mark.asyncio
async def test_conversation_rename_and_export(db, session_factory):
    app = FastAPI()
    app.include_router(chat_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    class FakeChatService:
        async def export_conversation(self, conversation_id, fmt):
            return "EXPORTED"

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    async with session_factory() as seed_db:
        conv = Conversation(title="Old title")
        seed_db.add(conv)
        await seed_db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.patch(f"/chat/{conv.id}", json={"title": "New title"})
        assert res.status_code == 200
        assert res.json()["title"] == "New title"

        res = await client.get(f"/chat/{conv.id}/export?format=markdown")
        assert res.status_code == 200
        assert res.text == "EXPORTED"
        assert "attachment" in res.headers["content-disposition"]

        res = await client.get(f"/chat/{conv.id}/export?format=json")
        assert res.status_code == 200
        assert res.text == "EXPORTED"

        assert (await client.patch("/chat/9999", json={"title": "x"})).status_code == 404


@pytest.mark.asyncio
async def test_link_documents_to_conversation(db, session_factory):
    app = FastAPI()
    app.include_router(document_router)

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    async with session_factory() as seed_db:
        conv = Conversation(title="conv")
        seed_db.add(conv)
        await seed_db.commit()
        doc_a = Document(name="a.pdf", file_path="/tmp/a.pdf")
        doc_b = Document(name="b.pdf", file_path="/tmp/b.pdf")
        seed_db.add_all([doc_a, doc_b])
        await seed_db.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [doc_a.id, doc_b.id]},
        )
        assert res.status_code == 200
        assert set(res.json()["linked"]) == {doc_a.id, doc_b.id}

        res = await client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [doc_a.id]},
        )
        assert res.json()["linked"] == []

        res = await client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [9999]},
        )
        assert res.status_code == 404

        listed = (await client.get(f"/documents/{conv.id}/documents")).json()
        assert {d["id"] for d in listed} == {doc_a.id, doc_b.id}
