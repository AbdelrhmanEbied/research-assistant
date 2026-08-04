from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.backend.database.repositories import (
    ConversationRepository,
    DocumentRepository,
)

UPLOAD_DIR = Path("data/uploads")


class DocumentService:
    def __init__(self, rag, db: Session):
        self.rag = rag
        self.db = db
        self.document_repo = DocumentRepository(db)
        self.conversation_repo = ConversationRepository(db)

    async def upload_document(
        self,
        conversation_id: int,
        file: UploadFile,
    ):
        conversation = self.conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found.")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        save_path = UPLOAD_DIR / f"{uuid4()}_{file.filename}"

        try:
            async with aiofiles.open(save_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            document = self.document_repo.create(
                name=file.filename,
                file_path=str(save_path),
            )

            self.document_repo.link_to_conversation(
                conversation_id,
                document.id,
            )

            await run_in_threadpool(self.rag.index, save_path)

            return document

        except Exception:
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            raise