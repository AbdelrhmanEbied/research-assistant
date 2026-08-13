from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.backend.database.database import get_db
from app.backend.database.repositories import DocumentRepository
from app.backend.schemas.document import DocumentDetailResponse, DocumentResponse
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
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{conversation_id}/documents", response_model=list[DocumentResponse])
def list_conversation_documents(
    conversation_id: int,
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = DocumentRepository(db)
    return repo.list_by_conversation(conversation_id)


class LinkDocumentsRequest(BaseModel):
    conversation_id: int
    document_ids: list[int] = Field(min_length=1)


@router.post("/link")
def link_documents(
    body: LinkDocumentsRequest,
    db: Session = Depends(get_db),  # noqa: B008
):
    """Attach existing documents to a conversation so retrieval can scope to them."""
    repo = DocumentRepository(db)
    existing_ids = {doc.id for doc in repo.list_all()}
    requested = set(body.document_ids)

    missing = requested - existing_ids
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Documents not found: {sorted(missing)}",
        )

    linked = repo.ensure_linked(body.conversation_id, list(requested))
    return {"linked": linked}



@router.get("/", response_model=list[DocumentDetailResponse])
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