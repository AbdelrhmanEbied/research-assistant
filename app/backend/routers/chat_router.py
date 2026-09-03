from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.database.database import get_db
from app.backend.database.repositories import ConversationRepository, MessageRepository
from app.backend.schemas.chat import ChatRequest, RegenerateRequest
from app.backend.schemas.conversation import (
    ConversationResponse,
    MessagesPage,
)
from app.backend.services.chat_service import ChatService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def get_checkpointer(request: Request):
    return request.app.state.checkpointer


async def get_chat_service(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ChatService:
    return ChatService(
        graph=request.app.state.graph,
        db=db,
        rag=request.app.state.rag,
    )


@router.post("/")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    return StreamingResponse(
        service.stream(request),
        media_type="text/plain",
    )


@router.post("/regenerate")
async def regenerate_response(
    request: RegenerateRequest,
    service: ChatService = Depends(get_chat_service),
):
    return StreamingResponse(
        service.regenerate(request),
        media_type="text/plain",
    )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    db: AsyncSession = Depends(get_db),
):
    repo = ConversationRepository(db)

    conversation = await repo.create()

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
    )


class RenameRequest(BaseModel):
    title: str


@router.patch("/{conversation_id}")
async def rename_conversation(
    conversation_id: int,
    body: RenameRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = ConversationRepository(db)
    conversation = await repo.update_title(conversation_id, body.title.strip())
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(id=conversation.id, title=conversation.title)


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    checkpointer=Depends(get_checkpointer),
):
    repo = ConversationRepository(db)

    conversation = await repo.get_by_id(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    await checkpointer.adelete_thread(str(conversation_id))

    await repo.delete(conversation_id)

    return {"message": "Conversation deleted successfully."}


@router.get("/list", response_model=list[ConversationResponse])
async def list_conversations(
    q: str | None = Query(default=None, max_length=200),
    db: AsyncSession = Depends(get_db),
):
    repo = ConversationRepository(db)
    if q and q.strip():
        return await repo.search(q.strip())
    return await repo.list_all()


@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: int,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
    service: ChatService = Depends(get_chat_service),
):
    try:
        content = await service.export_conversation(conversation_id, format)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if format == "json":
        media_type = "application/json"
        filename = f"conversation-{conversation_id}.json"
    else:
        media_type = "text/markdown"
        filename = f"conversation-{conversation_id}.md"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=MessagesPage,
)
async def list_messages(
    conversation_id: int,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    repo = MessageRepository(db)

    messages = await repo.list_by_conversation(
        conversation_id,
        limit=limit,
        offset=offset,
    )
    total = await repo.count_by_conversation(conversation_id)

    return MessagesPage(
        messages=messages,
        total=total,
        limit=limit,
        offset=offset,
    )
