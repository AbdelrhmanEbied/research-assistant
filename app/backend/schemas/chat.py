from enum import Enum

from pydantic import BaseModel, Field

from agent.agent_schemas import KnowledgeSource


class LLMConfig(BaseModel):
    model: str
    model_provider: str = "google_genai"
    api_key: str | None = None


class ModeOverride(str, Enum):
    AUTO = "auto"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXPLAIN = "explain"


class SourceOverride(str, Enum):
    AUTO = "auto"
    DOCUMENTS = "documents"
    WEB = "web"
    CHAT = "chat"


class RetrievalConfig(BaseModel):
    search_type: str | None = Field(
        default=None,
        description="One of 'hybrid', 'dense', 'sparse'.",
    )
    limit: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    search_depth: str | None = Field(
        default=None,
        description="Tavily web search depth: 'basic' or 'advanced'.",
    )


class ChatRequest(BaseModel):
    query: str
    conversation_id: int
    llm_config: LLMConfig | None = None
    mode: ModeOverride | None = None
    source: SourceOverride | None = None
    retrieval: RetrievalConfig | None = None


class RegenerateRequest(BaseModel):
    conversation_id: int
    llm_config: LLMConfig | None = None
    mode: ModeOverride | None = None
    source: SourceOverride | None = None
    retrieval: RetrievalConfig | None = None


class ChatResponse(BaseModel):
    response: str
    source: KnowledgeSource