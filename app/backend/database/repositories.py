from datetime import UTC, datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.backend.database.models import (
    Conversation,
    ConversationDocument,
    Document,
    Message,
)


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(title=title)

        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_by_id(self, conversation_id: int) -> Conversation | None:
        return self.db.get(Conversation, conversation_id)

    def delete(
        self,
        conversation_id: int,
    ) -> bool:
        conversation = self.get_by_id(conversation_id)

        if conversation is None:
            return False

        self.db.delete(conversation)

        self.db.commit()
        return True

    def list_all(self) -> list[Conversation]:
        return self.db.query(Conversation).order_by(Conversation.updated_at.desc()).all()

    def search(self, query: str) -> list[Conversation]:
        """Find conversations by title or message content (lightweight LIKE)."""
        like = f"%{query.strip()}%"
        matching_ids = self.db.query(Message.conversation_id).filter(Message.content.ilike(like))
        return (
            self.db.query(Conversation)
            .filter(
                or_(
                    Conversation.title.ilike(like),
                    Conversation.id.in_(matching_ids),
                )
            )
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def touch(self, conversation: Conversation) -> Conversation:
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_title(self, conversation_id: int, title: str):
        conversation = self.get_by_id(conversation_id=conversation_id)

        if conversation is None:
            return None

        conversation.title = title
        self.db.commit()
        self.db.refresh(conversation)
        return conversation


class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        self.db.add(message)

        conversation = self.db.get(Conversation, conversation_id)

        if conversation is not None:
            conversation.updated_at = datetime.now(tz=UTC)
            self.db.add(conversation)

        self.db.commit()
        self.db.refresh(message)

        return message

    def list_by_conversation(
        self,
        conversation_id: int,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Return messages newest-first (page 0 = latest messages)."""
        query = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
        )
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        return query.all()

    def list_for_history(
        self,
        conversation_id: int,
    ) -> list[Message]:
        """Return messages oldest-first, as required for LLM context."""
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )

    def count_by_conversation(
        self,
        conversation_id: int,
    ) -> int:
        return self.db.query(Message).filter(Message.conversation_id == conversation_id).count()

    def get_by_id(self, message_id: int) -> Message | None:
        return self.db.get(Message, message_id)

    def delete_by_id(self, message_id: int) -> bool:
        message = self.get_by_id(message_id)
        if message is None:
            return False
        self.db.delete(message)
        self.db.commit()
        return True

    def delete_after_id(self, conversation_id: int, after_id: int) -> int:
        """Delete messages in a conversation with ``id > after_id``."""
        deleted_count = (
            self.db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id > after_id,
            )
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return deleted_count

    def delete_by_conversation(
        self,
        conversation_id: int,
    ) -> int:

        deleted_count = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return deleted_count

    def update_metadata(
        self,
        message_id: int,
        metadata: dict | None,
    ) -> Message | None:
        message = self.db.get(Message, message_id)
        if message is None:
            return None
        message.extra = metadata or None
        self.db.commit()
        self.db.refresh(message)
        return message


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        name: str,
        file_path: str,
    ) -> Document:
        document = Document(
            name=name,
            file_path=file_path,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        return self.db.get(Document, document_id)

    def list_all(self) -> list[Document]:
        return self.db.query(Document).order_by(Document.id.desc()).all()

    def list_all_with_conversations(self) -> list[Document]:
        return (
            self.db.query(Document)
            .options(joinedload(Document.links).joinedload(ConversationDocument.conversation))
            .order_by(Document.id.desc())
            .all()
        )

    def link_to_conversation(self, conversation_id: int, document_id: int) -> ConversationDocument:
        link = ConversationDocument(
            conversation_id=conversation_id,
            document_id=document_id,
        )
        self.db.add(link)
        self.db.commit()
        self.db.refresh(link)
        return link

    def is_linked(self, conversation_id: int, document_id: int) -> bool:
        return (
            self.db.query(ConversationDocument)
            .filter(
                ConversationDocument.conversation_id == conversation_id,
                ConversationDocument.document_id == document_id,
            )
            .first()
            is not None
        )

    def ensure_linked(self, conversation_id: int, document_ids: list[int]) -> list[int]:
        """Link documents to a conversation, skipping already-linked ones."""
        linked = []
        for document_id in document_ids:
            if self.is_linked(conversation_id, document_id):
                continue
            self.link_to_conversation(conversation_id, document_id)
            linked.append(document_id)
        return linked

    def list_by_conversation(self, conversation_id: int) -> list[Document]:
        return (
            self.db.query(Document)
            .join(ConversationDocument)
            .filter(ConversationDocument.conversation_id == conversation_id)
            .all()
        )

    def delete(self, document_id: int) -> bool:
        document = self.get_by_id(document_id)
        if document is None:
            return False

        self.db.query(ConversationDocument).filter(
            ConversationDocument.document_id == document_id
        ).delete(synchronize_session=False)

        self.db.delete(document)
        self.db.commit()
        return True
