import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.graph import build_agent_graph
from agent.web_service import create_web_search_service
from app.backend.database.base import Base
from app.backend.database.database import engine
from app.backend.database.models import (
    Conversation,
    ConversationDocument,
    Document,
    Message,
)
from rag.rag_service import create_rag_service
from rag.reranker import Reranker
from telemetry import init_telemetry

logger = logging.getLogger("uvicorn.error")

PG_URL = "postgresql://postgres:postgres@localhost:5432/research_assistant"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application services...")

    try:
        logger.info("Initializing telemetry...")
        init_telemetry()

        logger.info("Initializing database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Loading shared Reranker model...")
        shared_reranker = Reranker(
            model="Xenova/ms-marco-MiniLM-L-12-v2",
        )

        logger.info("Initializing RAG service...")
        app.state.rag = create_rag_service(
            reranker=shared_reranker, db_path=None, collection_name="docs"
        )

        logger.info("Initializing Web Search service...")
        app.state.web_search = create_web_search_service(reranker=shared_reranker)

        logger.info("Initializing checkpointer...")
        checkpointer_serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("agent.agent_schemas", "PromptMode"),
                ("agent.agent_schemas", "KnowledgeSource"),
                ("rag.rag_schemas", "Context"),
                ("rag.rag_schemas", "KnowledgeResult"),
                ("rag.rag_schemas", "RetrievedDocuments"),
            ],
        )
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(
            PG_URL, serde=checkpointer_serde
        )
        app.state.checkpointer = await checkpointer_cm.__aenter__()
        app.state._checkpointer_cm = checkpointer_cm
        await app.state.checkpointer.setup()

        logger.info("Building agent graph...")
        app.state.graph = build_agent_graph(
            rag=app.state.rag,
            search_service=app.state.web_search,
            checkpointer=app.state.checkpointer,
        )

        logger.info("All application services initialized successfully.")

    except Exception as e:
        logger.exception("Startup failed: %s", e)
        raise

    yield

    logger.info("Shutting down application services...")

    if getattr(app.state, "rag", None):
        try:
            app.state.rag.qdrant_manager.close_client()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")

    if getattr(app.state, "_checkpointer_cm", None):
        try:
            await app.state._checkpointer_cm.__aexit__(None, None, None)
            logger.info("Checkpointer closed successfully.")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")

    await engine.dispose()

    app.state.rag = None
    app.state.web_search = None
    app.state.graph = None
    app.state.checkpointer = None

    logger.info("Application shutdown complete.")
