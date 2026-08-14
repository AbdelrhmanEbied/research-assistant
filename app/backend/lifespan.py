import logging
from contextlib import asynccontextmanager

import aiosqlite
from anyio import to_thread
from fastapi import FastAPI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.graph import build_agent_graph
from agent.web_service import create_web_search_service
from app.backend.database.base import Base
from app.backend.database.database import engine
from app.backend.database.migrations import ensure_schema_migrations
from app.backend.database.models import (
    Conversation,  # noqa: F401
    ConversationDocument,  # noqa: F401
    Document,  # noqa: F401
    Message,  # noqa: F401
)
from paths import data_path
from rag.rag_service import create_rag_service
from rag.reranker import Reranker
from telemetry import init_telemetry

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application services...")

    try:
        logger.info("Initializing telemetry...")
        init_telemetry()

        logger.info("Initializing database tables...")
        await to_thread.run_sync(lambda: Base.metadata.create_all(bind=engine))

        logger.info("Applying additive schema migrations...")
        await to_thread.run_sync(lambda: ensure_schema_migrations(engine))

        logger.info("Loading shared Reranker model...")
        shared_reranker = Reranker(
            model="Xenova/ms-marco-MiniLM-L-12-v2",
        )

        logger.info("Initializing RAG service...")
        app.state.rag = create_rag_service(
            reranker=shared_reranker, db_path=str(data_path("qdrant_db")), collection_name="docs"
        )

        logger.info("Initializing Web Search service...")
        app.state.web_search = create_web_search_service(reranker=shared_reranker)

        logger.info("Initializing checkpointer...")
        checkpointer_conn = await aiosqlite.connect(str(data_path("checkpoints.db")))
        checkpointer_serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("agent.agent_schemas", "PromptMode"),
                ("agent.agent_schemas", "KnowledgeSource"),
                ("rag.rag_schemas", "Context"),
                ("rag.rag_schemas", "KnowledgeResult"),
                ("rag.rag_schemas", "RetrievedDocuments"),
            ],
        )
        app.state.checkpointer = AsyncSqliteSaver(checkpointer_conn, serde=checkpointer_serde)
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
        if "checkpointer_conn" in locals():
            await checkpointer_conn.close()
        raise

    yield

    logger.info("Shutting down application services...")

    if getattr(app.state, "rag", None):
        try:
            app.state.rag.qdrant_manager.close_client()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")

    if getattr(app.state, "checkpointer", None):
        try:
            await app.state.checkpointer.conn.close()
            logger.info("Checkpointer closed successfully.")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")

    app.state.rag = None
    app.state.web_search = None
    app.state.graph = None
    app.state.checkpointer = None

    logger.info("Application shutdown complete.")
