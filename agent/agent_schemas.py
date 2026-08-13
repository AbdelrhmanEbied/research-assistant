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

def replace_history(
    _current: list[ChatMessage],
    new: list[ChatMessage],
) -> list[ChatMessage]:
    """Replace rather than accumulate history.

    The complete conversation history is supplied from the local database on
    every request, so appending to the checkpointer's stale history would
    duplicate turns. Replacing keeps the graph state consistent with what is
    actually stored.
    """
    return new

class AgentState(TypedDict):
    history: Annotated[list[ChatMessage], replace_history]
    query: str
    mode: PromptMode | None
    source: KnowledgeSource | None
    knowledge_result: KnowledgeResult | None
    response: str | None
    conversation_id: str | None
    llm_config: dict | None
    sources: list[dict]
    mode_override: str | None
    source_override: str | None
    retrieval_config: dict | None

