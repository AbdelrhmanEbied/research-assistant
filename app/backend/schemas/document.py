from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    file_path: str


class DocumentConversationResponse(BaseModel):
    id: int
    title: str | None


class DocumentDetailResponse(DocumentResponse):
    conversations: list[DocumentConversationResponse] = []
