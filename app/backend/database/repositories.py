from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.backend.database.models import (
    Conversation,
    ConversationDocument,
    Document,
    Message,
)


class ConversationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(title=title)

        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def get_by_id(self, conversation_id: int) -> Conversation | None:
        return await self.db.get(Conversation, conversation_id)

    async def delete(
        self,
        conversation_id: int,
    ) -> bool:
        conversation = await self.get_by_id(conversation_id)

        if conversation is None:
            return False

        await self.db.delete(conversation)

        await self.db.commit()
        return True

    async def list_all(self) -> list[Conversation]:
        result = await self.db.execute(
            select(Conversation).order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    async def search(self, query: str) -> list[Conversation]:
        like = f"%{query.strip()}%"
        matching_ids = select(Message.conversation_id).filter(Message.content.ilike(like))
        result = await self.db.execute(
            select(Conversation)
            .filter(
                or_(
                    Conversation.title.ilike(like),
                    Conversation.id.in_(matching_ids),
                )
            )
            .order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    async def touch(self, conversation: Conversation) -> Conversation:
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def update_title(self, conversation_id: int, title: str):
        conversation = await self.get_by_id(conversation_id=conversation_id)

        if conversation is None:
            return None

        conversation.title = title
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation


class MessageRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def add_message(
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

        conversation = await self.db.get(Conversation, conversation_id)

        if conversation is not None:
            conversation.updated_at = datetime.now(tz=UTC)
            self.db.add(conversation)

        await self.db.commit()
        await self.db.refresh(message)

        return message

    async def list_by_conversation(
        self,
        conversation_id: int,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        query = (
            select(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
        )
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def list_for_history(
        self,
        conversation_id: int,
    ) -> list[Message]:
        result = await self.db.execute(
            select(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        return list(result.scalars().all())

    async def count_by_conversation(
        self,
        conversation_id: int,
    ) -> int:
        result = await self.db.execute(
            select(Message).filter(Message.conversation_id == conversation_id)
        )
        return len(result.scalars().all())

    async def get_by_id(self, message_id: int) -> Message | None:
        return await self.db.get(Message, message_id)

    async def delete_by_id(self, message_id: int) -> bool:
        message = await self.get_by_id(message_id)
        if message is None:
            return False
        await self.db.delete(message)
        await self.db.commit()
        return True

    async def delete_after_id(self, conversation_id: int, after_id: int) -> int:
        result = await self.db.execute(
            select(Message).filter(
                Message.conversation_id == conversation_id,
                Message.id > after_id,
            )
        )
        messages = result.scalars().all()
        for message in messages:
            await self.db.delete(message)
        await self.db.commit()
        return len(messages)

    async def delete_by_conversation(
        self,
        conversation_id: int,
    ) -> int:
        result = await self.db.execute(
            select(Message).filter(Message.conversation_id == conversation_id)
        )
        messages = result.scalars().all()
        for message in messages:
            await self.db.delete(message)
        await self.db.commit()
        return len(messages)

    async def update_metadata(
        self,
        message_id: int,
        metadata: dict | None,
    ) -> Message | None:
        message = await self.db.get(Message, message_id)
        if message is None:
            return None
        message.extra = metadata or None
        await self.db.commit()
        await self.db.refresh(message)
        return message


class DocumentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        name: str,
        file_path: str,
    ) -> Document:
        document = Document(
            name=name,
            file_path=file_path,
        )
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)
        return document

    async def get_by_id(self, document_id: int) -> Document | None:
        return await self.db.get(Document, document_id)

    async def list_all(self) -> list[Document]:
        result = await self.db.execute(select(Document).order_by(Document.id.desc()))
        return list(result.scalars().all())

    async def list_all_with_conversations(self) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .options(joinedload(Document.links).joinedload(ConversationDocument.conversation))
            .order_by(Document.id.desc())
        )
        return list(result.scalars().all())

    async def link_to_conversation(self, conversation_id: int, document_id: int) -> ConversationDocument:
        link = ConversationDocument(
            conversation_id=conversation_id,
            document_id=document_id,
        )
        self.db.add(link)
        await self.db.commit()
        await self.db.refresh(link)
        return link

    async def is_linked(self, conversation_id: int, document_id: int) -> bool:
        result = await self.db.execute(
            select(ConversationDocument).filter(
                ConversationDocument.conversation_id == conversation_id,
                ConversationDocument.document_id == document_id,
            )
        )
        return result.scalars().first() is not None

    async def ensure_linked(self, conversation_id: int, document_ids: list[int]) -> list[int]:
        linked = []
        for document_id in document_ids:
            if await self.is_linked(conversation_id, document_id):
                continue
            await self.link_to_conversation(conversation_id, document_id)
            linked.append(document_id)
        return linked

    async def list_by_conversation(self, conversation_id: int) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .join(ConversationDocument)
            .filter(ConversationDocument.conversation_id == conversation_id)
        )
        return list(result.scalars().all())

    async def delete(self, document_id: int) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False

        result = await self.db.execute(
            select(ConversationDocument).filter(ConversationDocument.document_id == document_id)
        )
        links = result.scalars().all()
        for link in links:
            await self.db.delete(link)

        await self.db.delete(document)
        await self.db.commit()
        return True
