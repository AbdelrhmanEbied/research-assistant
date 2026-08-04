from fastapi import FastAPI, status

from app.backend.lifespan import lifespan
from app.backend.routers.chat_router import router as chat_router
from app.backend.routers.document_router import router as document_router

app = FastAPI(
    title = "LangGraph Test",
    lifespan=lifespan,
)


app.include_router(chat_router)
app.include_router(document_router)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


