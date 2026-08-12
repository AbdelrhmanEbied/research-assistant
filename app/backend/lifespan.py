import logging
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.graph import build_agent_graph
from agent.web_service import create_web_search_service
from app.backend.database.base import Base
from app.backend.database.database import engine
from app.backend.database.models import (
    Conversation,  # noqa: F401
    ConversationDocument,  # noqa: F401
    Document,  # noqa: F401
    Message,  # noqa: F401
)
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

        logger.info("Loading shared Reranker model...")
        shared_reranker = Reranker(
            model="Xenova/ms-marco-MiniLM-L-12-v2",
        )

        logger.info("Initializing RAG service...")
        app.state.rag = create_rag_service(reranker=shared_reranker, db_path="./qdrant_db", collection_name="docs")

        logger.info("Initializing Web Search service...")
        app.state.web_search = create_web_search_service(reranker=shared_reranker)

        logger.info("Initializing checkpointer...")
        checkpointer_cm = AsyncSqliteSaver.from_conn_string("checkpoints.db")
        app.state.checkpointer = await checkpointer_cm.__aenter__()

        logger.info("Building agent graph...")
        app.state.graph = build_agent_graph(
            rag=app.state.rag,
            search_service=app.state.web_search,
            checkpointer=app.state.checkpointer,
        )

        logger.info("All application services initialized successfully.")

    except Exception as e:
        logger.exception("Startup failed: %s", e)
        if "checkpointer_cm" in locals():
            await checkpointer_cm.__aexit__(None, None, None)
        raise

    yield

    logger.info("Shutting down application services...")

    if getattr(app.state, "rag", None):
        try:
            app.state.rag.qdrant_manager.close_client()
            logger.info("Qdrant client closed.")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")

    if "checkpointer_cm" in locals():
        try:
            await checkpointer_cm.__aexit__(None, None, None)
            logger.info("Checkpointer closed successfully.")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")

    app.state.rag = None
    app.state.web_search = None
    app.state.graph = None
    app.state.checkpointer = None

    logger.info("Application shutdown complete.")
