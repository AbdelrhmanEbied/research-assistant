from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.backend.database.database import get_db
from app.backend.database.repositories import DocumentRepository
from app.backend.services.document_service import DocumentService

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def get_document_service(
    request: Request,
    db: Session = Depends(get_db),  # noqa: B008
):
    return DocumentService(
        rag=request.app.state.rag,
        db=db,
    )


@router.post("/upload")
async def upload_document(
    conversation_id: int  = Form(...),
    file: UploadFile = File(...),  # noqa: B008
    service: DocumentService = Depends(get_document_service),  # noqa: B008
):
    return await service.upload_document(
        conversation_id=conversation_id,
        file=file,
    )


@router.get("/{conversation_id}/documents")
def list_conversation_documents(
    conversation_id: int,
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = DocumentRepository(db)
    return repo.list_by_conversation(conversation_id)

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = DocumentRepository(db)

    document = repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document.file_path
    path = Path(file_path)

    if path.exists() and path.is_file():
        path.unlink()

    deleted = repo.delete(document_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"message": "Document deleted successfully."}