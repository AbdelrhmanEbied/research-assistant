from datetime import datetime

from pydantic import BaseModel


class ConversationResponse(BaseModel):
    id: int
    title: str | None

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime