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
from app.backend.schemas.document import DocumentResponse
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
    try:
        return await service.upload_document(
            conversation_id=conversation_id,
            file=file,
        )
    except ValueError as e:
        # bad file type or missing conversation - client's mistake,
        # not the server's, so it's a 400 not a 500
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{conversation_id}/documents", response_model=list[DocumentResponse])
def list_conversation_documents(
    conversation_id: int,
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = DocumentRepository(db)
    return repo.list_by_conversation(conversation_id)


# every document across every conversation, not scoped to one chat
@router.get("/", response_model=list[DocumentResponse])
def list_all_documents(
    service: DocumentService = Depends(get_document_service),  # noqa: B008
):
    return service.list_all_documents()


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    service: DocumentService = Depends(get_document_service),  # noqa: B008
):
    try:
        service.delete_document(document_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None

    return {"message": "Document deleted successfully."}