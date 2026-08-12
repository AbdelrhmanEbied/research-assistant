from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.backend.database.database import get_db
from app.backend.database.repositories import ConversationRepository, MessageRepository
from app.backend.schemas.chat import ChatRequest
from app.backend.schemas.conversation import ConversationResponse, MessageResponse
from app.backend.services.chat_service import ChatService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

def get_checkpointer(request: Request):
    return request.app.state.checkpointer

def get_chat_service(request: Request, db: Session = Depends(get_db)) -> ChatService:  # noqa: B008
    return ChatService(
        graph=request.app.state.graph,
        checkpointer=request.app.state.checkpointer,
        db=db,
        rag=request.app.state.rag,
    )


@router.post("/")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),  # noqa: B008
):
    return StreamingResponse(
        service.stream(request),
        media_type="text/plain",
    )


@router.post("/conversations",response_model= ConversationResponse)

def create_conversation(
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = ConversationRepository(db)

    conversation = repo.create()

    return ConversationResponse(
        id = conversation.id,
        title = conversation.title,
    )



@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),  # noqa: B008
    checkpointer=Depends(get_checkpointer),  # noqa: B008
):
    repo = ConversationRepository(db)

    conversation = repo.get_by_id(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    await checkpointer.adelete_thread(str(conversation_id))

    repo.delete(conversation_id)

    return {
        "message": "Conversation deleted successfully."
    }


@router.get("/list", response_model=list[ConversationResponse])
def list_conversations(
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = ConversationRepository(db)
    return repo.list_all()


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def list_messages(
    conversation_id: int,
    db: Session = Depends(get_db),  # noqa: B008
):
    repo = MessageRepository(db)

    return repo.list_by_conversation(conversation_id)