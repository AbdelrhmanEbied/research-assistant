from enum import Enum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from rag.rag_schemas import KnowledgeResult


class PromptMode(Enum):
    CHAT = "chat"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXPLAIN = "explain"


class KnowledgeSource(Enum):
    NONE = "none"
    RAG = "rag"
    WEB = "web"



class RouteQuery(BaseModel):
    mode: PromptMode = Field(
        description="""
        The assistant's execution mode.

        Choose the mode that best represents the user's primary intent:
          - CHAT: Casual conversation
          - SUMMARIZE: Condensing content
          - COMPARE: Identifying similarities and differences
          - EXPLAIN: Teaching or clarifying a concept
        """
    )

    source: KnowledgeSource = Field(
        description="""
        The knowledge source required to answer the user's request.

        - NONE: No external retrieval needed; answer directly.
        - RAG: Answer from the indexed knowledge base (PDFs, internal docs).
        - WEB: Request requires live or internet-based information.
        """
    )
class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str

def append_history(
    current: list[ChatMessage],
    new: list[ChatMessage],
) -> list[ChatMessage]:
    return current + new

class AgentState(TypedDict):
    history: Annotated[list[ChatMessage], append_history]
    query: str
    mode: PromptMode | None
    source: KnowledgeSource | None
    knowledge_result: KnowledgeResult | None
    response: str | None
    conversation_id: str | None

