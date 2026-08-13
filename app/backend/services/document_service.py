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
from app.backend.schemas.document import (
    DocumentConversationResponse,
    DocumentDetailResponse,
)
from telemetry import clear_request_tracking, start_request_tracking

UPLOAD_DIR = Path("data/uploads")


class DocumentService:
    def __init__(self, rag, db: Session):
        self.rag = rag
        self.db = db
        self.document_repo = DocumentRepository(db)
        self.conversation_repo = ConversationRepository(db)

    def _embedding_model_name(self) -> str | None:
        try:
            return getattr(self.rag.embedder.dense_model, "model_name", None)
        except Exception:
            return None

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
        tracker = start_request_tracking(
            route="/documents/upload",
            conversation_id=conversation_id,
            embedding_model=self._embedding_model_name(),
        )

        try:
            conversation = self.conversation_repo.get_by_id(conversation_id)
            if conversation is None:
                raise ValueError("Conversation not found.")


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

                async with tracker.timed("index_latency_ms"):
                    await run_in_threadpool(
                        self.rag.index,
                        save_path,
                        {
                            "document_id": str(document.id),
                            "conversation_id": str(conversation_id),
                            "name": document.name,
                        },
                    )

            except Exception:
                if save_path.exists():
                    save_path.unlink(missing_ok=True)
                if document is not None:
                    self.document_repo.delete(document.id)
                raise

            tracker.finish(success=True)
            return document

        except Exception as exc:
            tracker.finish(success=False, error_type=type(exc).__name__)
            raise
        finally:
            clear_request_tracking()

    def list_all_documents(self) -> list[DocumentDetailResponse]:
        documents = self.document_repo.list_all_with_conversations()
        return [
            DocumentDetailResponse(
                id=document.id,
                name=document.name,
                file_path=document.file_path,
                conversations=[
                    DocumentConversationResponse(
                        id=link.conversation.id,
                        title=link.conversation.title,
                    )
                    for link in document.links
                ],
            )
            for document in documents
        ]

    def delete_document(self, document_id: int):
        document = self.document_repo.get_by_id(document_id)
        if document is None:
            raise ValueError("Document not found.")

        file_path = Path(document.file_path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()


        self.rag.qdrant_manager.delete_document(document_id=str(document_id))

        self.document_repo.delete(document_id)