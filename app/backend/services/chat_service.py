import re
from collections.abc import AsyncGenerator

from sqlalchemy.orm import Session

from agent.llms import llm as generation_llm
from app.backend.database.repositories import ConversationRepository, MessageRepository
from app.backend.schemas.chat import ChatRequest
from telemetry import clear_request_tracking, start_request_tracking


class ChatService:
    def __init__(self, graph, checkpointer, db: Session, rag=None):
        self.graph = graph
        self.checkpointer = checkpointer
        self.db = db
        self.rag = rag
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)

    def _embedding_model_name(self) -> str | None:
        try:
            return getattr(self.rag.embedder.dense_model, "model_name", None)
        except Exception:
            return None

    def _generation_model_name(self) -> str | None:
        return (
            getattr(generation_llm, "model_name", None)
            or getattr(generation_llm, "model", None)
        )

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

        tracker = start_request_tracking(
            route="/chat/",
            conversation_id=request.conversation_id,
            model=self._generation_model_name(),
            embedding_model=self._embedding_model_name(),
        )

        assistant_buffer: list[str] = []

        state = {
            "query": request.query,
            "conversation_id": str(request.conversation_id),
        }

        config = {
            "configurable": {
                "thread_id": str(request.conversation_id),
            }
        }

        try:
            async with tracker.span(
                "chat_request",
                span_type="AGENT",
                latency_metric="agent_latency_ms",
            ):
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

            tracker.finish(success=True)

        except Exception as exc:
            tracker.finish(success=False, error_type=type(exc).__name__)
            raise
        finally:
            clear_request_tracking()

        full_answer = "".join(assistant_buffer).strip()
        if full_answer:
            self.message_repo.add_message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=full_answer,
            )