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

    def _supported_extensions(self) -> list[str]:
        return sorted(self.rag.loader.loaders)

    def _validate_extension(self, filename: str) -> None:
        suffix = Path(filename or "").suffix.lower()
        supported = self._supported_extensions()
        if suffix not in supported:
            raise ValueError(
                f"Unsupported file type '{suffix or filename}'. "
                f"Supported types: {', '.join(supported)}"
            )

    async def upload_document(
        self,
        conversation_id: int,
        file: UploadFile,
    ):
        conversation = self.conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise ValueError("Conversation not found.")

        # catch unsupported types (e.g. .mp4) here, not mid-index -
        # otherwise the loader just blows up and it comes out as a 500
        self._validate_extension(file.filename)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        save_path = UPLOAD_DIR / f"{uuid4()}_{file.filename}"
        document = None

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

            # tag every chunk with the SQL id so delete_document() can
            # find and purge exactly these vectors later, not all of them
            await run_in_threadpool(
                self.rag.index,
                save_path,
                {"document_id": str(document.id)},
            )

            return document

        except Exception:
            # roll back the DB row too, not just the file - otherwise a
            # failed index leaves an orphaned document behind forever
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            if document is not None:
                self.document_repo.delete(document.id)
            raise

    def list_all_documents(self):
        return self.document_repo.list_all()

    def delete_document(self, document_id: int):
        document = self.document_repo.get_by_id(document_id)
        if document is None:
            raise ValueError("Document not found.")

        file_path = Path(document.file_path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()

        # skip this and the vectors stay searchable forever - "deleting"
        # a document wouldn't actually stop it answering questions
        self.rag.qdrant_manager.delete_document(document_id=str(document_id))

        self.document_repo.delete(document_id)