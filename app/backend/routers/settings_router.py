from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from settings import get_settings_store

router = APIRouter(
    prefix="/settings",
    tags=["Settings"],
)


class LLMSettingsUpdate(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    model_provider: str = Field(min_length=1, max_length=100)


class ApiKeyUpdate(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    api_key: str | None = Field(default=None, max_length=500)


class RetrievalSettingsUpdate(BaseModel):
    search_type: str | None = Field(default=None, pattern="^(hybrid|dense|sparse)$")
    limit: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    search_depth: str | None = Field(default=None, pattern="^(basic|advanced)$")


@router.get("/")
def get_settings():
    """Return non-secret settings. API key presence is a boolean only."""
    return get_settings_store().public_dict()


@router.put("/llm")
def update_llm(body: LLMSettingsUpdate):
    get_settings_store().set_llm(body.model.strip(), body.model_provider.strip())
    return {"ok": True}


@router.put("/api-keys")
def update_api_key(body: ApiKeyUpdate):
    """Store an API key for a provider. The value is never returned or logged."""
    if body.api_key and not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be blank.")
    get_settings_store().set_api_key(body.provider.strip(), body.api_key)
    return {"ok": True}


@router.put("/retrieval")
def update_retrieval(body: RetrievalSettingsUpdate):
    get_settings_store().set_retrieval(
        search_type=body.search_type,
        limit=body.limit,
        rerank=body.rerank,
        search_depth=body.search_depth,
    )
    return {"ok": True}