import re
from collections.abc import AsyncGenerator

from sqlalchemy.orm import Session

from app.backend.database.repositories import ConversationRepository, MessageRepository
from app.backend.schemas.chat import ChatRequest


class ChatService:
    def __init__(self, graph, checkpointer, db: Session):
        self.graph = graph
        self.checkpointer = checkpointer
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)

    async def generate_title(self, query: str, max_length: int = 50) -> str:
        title = re.sub(r"\s+", " ", query.strip())
        title = title.rstrip(".,!?;:")

        if not title:
            return "New Chat"

        if len(title) <= max_length:
            return title

        truncated = title[:max_length]

        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]

        return truncated + "..."

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str]:
        conversation = self.conversation_repo.get_by_id(request.conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {request.conversation_id} not found")

        if not conversation.title:
            self.conversation_repo.update_title(
                conversation.id,
                await self.generate_title(request.query)
            )


        self.message_repo.add_message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.query,
        )

        assistant_buffer: list[str] = []

        state = {
            "query": request.query,
        }

        config = {
            "configurable": {
                "thread_id": str(request.conversation_id),
            }
        }

        async for event in self.graph.astream_events(
            state,
            config=config,
            version="v2",
        ):
            if event["event"] != "on_chat_model_stream":
                continue

            if event.get("metadata", {}).get("langgraph_node") != "generate_answer":
                continue

            chunk = event["data"]["chunk"]

            if not chunk.content:
                continue

            for block in chunk.content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        assistant_buffer.append(text)
                        yield text

        full_answer = "".join(assistant_buffer).strip()
        if full_answer:
            self.message_repo.add_message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=full_answer,
            )

