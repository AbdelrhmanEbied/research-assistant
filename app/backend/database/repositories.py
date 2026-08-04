from datetime import UTC, datetime

from sqlalchemy.orm import Session

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

    def get_by_id(
            self,
            conversation_id: int
    ) -> Conversation | None:
        return self.db.get(Conversation,conversation_id)

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
        return(
            self.db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    def touch(
            self,
            conversation:Conversation
    ) -> Conversation:
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def update_title(
        self,
        conversation_id: int,
        title: str 
    ):
        conversation = self.get_by_id(conversation_id=conversation_id)

        if conversation is None:
            return None

        conversation.title = title
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

class MessageRepository:
    def __init__(
            self,
            db:Session
    ):
        self.db = db

    def add_message(
            self,
            conversation_id : int,
            role: str,
            content: str,
    ) -> Message:

        message = Message(
            conversation_id = conversation_id,
            role = role,
            content = content,
        )
        self.db.add(message)

        conversation = self.db.get(Conversation,conversation_id)

        if conversation is not None:
            conversation.updated_at = datetime.now(tz=UTC)
            self.db.add(conversation)

        self.db.commit()
        self.db.refresh(message)

        return message

    def list_by_conversation(
            self,
            conversation_id: int,
    ) -> list[Message]:

        return(
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(),Message.id.asc())
            .all()
        )

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
        return (
            self.db.query(Document)
            .order_by(Document.uploaded_at.desc())
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

        self.db.delete(document)
        self.db.commit()
        return True



    