from pydantic import BaseModel

from agent.agent_schemas import KnowledgeSource


class ChatRequest(BaseModel):
    query: str
    conversation_id: int
class ChatResponse(BaseModel):
    response: str
    source: KnowledgeSource