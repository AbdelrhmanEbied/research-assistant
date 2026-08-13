from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.backend.database.base import Base
from app.backend.database.database import get_db
from app.backend.database.models import Conversation, ConversationDocument, Document, Message
from app.backend.routers.chat_router import get_chat_service, router as chat_router
from app.backend.routers.document_router import get_document_service, router as document_router


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = testing_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def chat_app(db_session):
    app = FastAPI()
    app.include_router(chat_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    class FakeChatService:
        async def stream(self, request):
            yield "hello"
            yield " world"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    return app


def test_stream_chat(chat_app):
    with TestClient(chat_app) as client:
        response = client.post(
            "/chat/",
            json={"query": "hi", "conversation_id": 1},
        )
    assert response.status_code == 200
    assert response.text == "hello world"


def test_conversation_crud_and_message_pagination(db_session):
    app = FastAPI()
    app.include_router(chat_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            return None

    app.state.checkpointer = FakeCheckpointer()

    with TestClient(app) as client:
        created = client.post("/chat/conversations").json()
        conversation_id = created["id"]

        assert client.get("/chat/list").json()[0]["id"] == conversation_id

        base = datetime.now(UTC)
        for i in range(5):
            db_session.add(
                Message(
                    conversation_id=conversation_id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"msg-{i}",
                    created_at=base + timedelta(seconds=i),
                )
            )
        db_session.commit()

        page = client.get(f"/chat/{conversation_id}/messages?limit=2&offset=0").json()
        assert page["total"] == 5
        assert page["limit"] == 2
        assert page["offset"] == 0
        assert [m["content"] for m in page["messages"]] == ["msg-4", "msg-3"]

        page2 = client.get(f"/chat/{conversation_id}/messages?limit=2&offset=2").json()
        assert [m["content"] for m in page2["messages"]] == ["msg-2", "msg-1"]

        assert client.delete(f"/chat/{conversation_id}").status_code == 200
        assert client.delete(f"/chat/{conversation_id}").status_code == 404


def test_conversation_delete_uses_checkpointer(db_session):
    app = FastAPI()
    app.include_router(chat_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    deleted_threads = []

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            deleted_threads.append(thread_id)

    app.state.checkpointer = FakeCheckpointer()

    with TestClient(app) as client:
        conversation_id = client.post("/chat/conversations").json()["id"]
        assert client.delete(f"/chat/{conversation_id}").status_code == 200
        assert deleted_threads == [str(conversation_id)]


@pytest.fixture
def document_app(db_session):
    app = FastAPI()
    app.include_router(document_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    class FakeDocumentService:
        def __init__(self):
            self.uploaded = []
            self.deleted = []

        async def upload_document(self, conversation_id, file):
            self.uploaded.append((conversation_id, file.filename))
            return {"id": 1, "name": file.filename}

        def list_all_documents(self):
            return [
                {
                    "id": 1,
                    "name": "a.pdf",
                    "file_path": "/tmp/a.pdf",
                    "conversations": [],
                }
            ]

        def delete_document(self, document_id):
            self.deleted.append(document_id)

    fake = FakeDocumentService()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_document_service] = lambda: fake
    return app, fake


def test_document_upload_and_delete(document_app):
    app, fake = document_app
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            data={"conversation_id": 1},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "notes.txt"
        assert fake.uploaded == [(1, "notes.txt")]

        assert client.get("/documents/").json()[0]["name"] == "a.pdf"

        assert client.delete("/documents/1").status_code == 200
        assert fake.deleted == [1]


def test_document_list_by_conversation(db_session):
    app = FastAPI()
    app.include_router(document_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    conversation = Conversation(title="conv")
    db_session.add(conversation)
    db_session.commit()

    doc = Document(name="a.pdf", file_path="/tmp/a.pdf")
    db_session.add(doc)
    db_session.commit()

    from app.backend.database.models import ConversationDocument

    db_session.add(
        ConversationDocument(conversation_id=conversation.id, document_id=doc.id)
    )
    db_session.commit()

    with TestClient(app) as client:
        response = client.get(f"/documents/{conversation.id}/documents")
        assert response.status_code == 200
        assert response.json()[0]["name"] == "a.pdf"


def test_conversation_search_by_title_and_content(db_session):
    app = FastAPI()
    app.include_router(chat_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    conv_a = Conversation(title="Quantum computing notes")
    conv_b = Conversation(title="Cooking")
    db_session.add_all([conv_a, conv_b])
    db_session.commit()

    db_session.add(
        Message(conversation_id=conv_a.id, role="user", content="Tell me about qubits")
    )
    db_session.add(
        Message(conversation_id=conv_b.id, role="user", content="How do I make pasta?")
    )
    db_session.commit()

    with TestClient(app) as client:
        # matches on title
        titles = [c["title"] for c in client.get("/chat/list?q=quantum").json()]
        assert titles == ["Quantum computing notes"]

        # matches on message content
        titles = [c["title"] for c in client.get("/chat/list?q=qubits").json()]
        assert titles == ["Quantum computing notes"]

        # no matches
        assert client.get("/chat/list?q=zzzzz").json() == []


def test_conversation_rename_and_export(db_session):
    app = FastAPI()
    app.include_router(chat_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    class FakeChatService:
        async def export_conversation(self, conversation_id, fmt):
            return "EXPORTED"

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    conv = Conversation(title="Old title")
    db_session.add(conv)
    db_session.commit()

    with TestClient(app) as client:
        res = client.patch(f"/chat/{conv.id}", json={"title": "New title"})
        assert res.status_code == 200
        assert res.json()["title"] == "New title"

        res = client.get(f"/chat/{conv.id}/export?format=markdown")
        assert res.status_code == 200
        assert res.text == "EXPORTED"
        assert "attachment" in res.headers["content-disposition"]

        res = client.get(f"/chat/{conv.id}/export?format=json")
        assert res.status_code == 200
        assert res.text == "EXPORTED"

        assert client.patch("/chat/9999", json={"title": "x"}).status_code == 404


def test_link_documents_to_conversation(db_session):
    app = FastAPI()
    app.include_router(document_router)

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    conv = Conversation(title="conv")
    db_session.add(conv)
    db_session.commit()
    doc_a = Document(name="a.pdf", file_path="/tmp/a.pdf")
    doc_b = Document(name="b.pdf", file_path="/tmp/b.pdf")
    db_session.add_all([doc_a, doc_b])
    db_session.commit()

    with TestClient(app) as client:
        res = client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [doc_a.id, doc_b.id]},
        )
        assert res.status_code == 200
        assert set(res.json()["linked"]) == {doc_a.id, doc_b.id}

        # idempotent
        res = client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [doc_a.id]},
        )
        assert res.json()["linked"] == []

        # unknown document
        res = client.post(
            "/documents/link",
            json={"conversation_id": conv.id, "document_ids": [9999]},
        )
        assert res.status_code == 404

        listed = client.get(f"/documents/{conv.id}/documents").json()
        assert {d["id"] for d in listed} == {doc_a.id, doc_b.id}