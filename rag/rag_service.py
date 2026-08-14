import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient, models

from agent.agent_schemas import ChatMessage, PromptMode
from agent.prompts import SYSTEM_PROMPT
from rag.builders import ContextBuilder, PromptBuilder
from rag.chunker import DocumentChunker
from rag.embedder import DocumentEmbedder
from rag.loader import DocumentLoader
from rag.qdrant_manager import QDrantManager
from rag.rag_schemas import KnowledgeResult, RetrievedDocuments, SearchType
from rag.reranker import Reranker
from rag.retriever import Retriever
from telemetry import get_current_tracker

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


class RAGService:
    def __init__(
        self,
        loader,
        chunker,
        embedder,
        qdrant_manager,
        retriever,
        reranker,
        context_builder,
        prompt_builder,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.qdrant_manager = qdrant_manager
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder

    def load_documents(self, sources):
        return self.loader.load(sources)

    def chunk_documents(self, documents):
        return self.chunker.chunk(documents)

    def embed_chunks(self, chunks):
        return self.embedder.embed_documents(chunks)

    def upsert_chunks(self, embedded_chunks):
        return self.qdrant_manager.upsert_documents(embedded_chunks)

    def index(self, sources, metadata: dict | None = None):
        documents = self.load_documents(sources)
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        chunks = self.chunk_documents(documents)
        embedded_chunks = self.embed_chunks(chunks)
        return self.upsert_chunks(embedded_chunks)

    def retrieve(
        self,
        query,
        limit,
        search_type: str | SearchType | None = None,
        conversation_id: str | None = None,
    ):
        if isinstance(search_type, str):
            search_type = SearchType(search_type)
        return self.retriever.retrieve(
            query=query, limit=limit, search_type=search_type, conversation_id=conversation_id
        )

    def rerank(self, query, documents, top_k=5):
        return self.reranker.rerank(query=query, documents=documents, top_k=top_k)

    def build_context(self, documents):
        return self.context_builder.build(documents=documents)

    def build_prompt(self, question, context, history: list[ChatMessage], mode: PromptMode | str):
        return self.prompt_builder.build(
            question=question, context=context, mode=mode, history=history
        )

    def prepare(
        self,
        *,
        query: str,
        mode: PromptMode | str,
        history: list[ChatMessage],
        retrieve: bool = True,
        rerank: bool = True,
        limit: int = 10,
        rerank_top_k: int = 5,
        search_type: str | SearchType | None = None,
        conversation_id: str | None = None,
    ) -> KnowledgeResult:

        retrieved_documents: list[RetrievedDocuments] = []
        reranked_documents: list[RetrievedDocuments] = []

        tracker = get_current_tracker()

        with tracker.span(
            "rag_prepare",
            span_type="RAG",
            latency_metric="rag_latency_ms",
        ):
            if retrieve:
                retrieved_documents = self.retrieve(
                    query=query,
                    limit=limit,
                    search_type=search_type,
                    conversation_id=conversation_id,
                )

                if rerank:
                    reranked_documents = self.rerank(
                        query=query,
                        documents=retrieved_documents,
                        top_k=rerank_top_k,
                    )
                else:
                    reranked_documents = retrieved_documents

            context = self.build_context(documents=reranked_documents)

            prompt = self.build_prompt(
                mode=mode,
                question=query,
                context=context,
                history=history,
            )

        if retrieve:
            tracker.add_metric("retrieved_documents", len(retrieved_documents))
            tracker.add_metric("reranked_documents", len(reranked_documents))

        return KnowledgeResult(
            query=query,
            retrieved_documents=retrieved_documents,
            reranked_documents=reranked_documents,
            context=context,
            prompt=prompt,
        )


def create_rag_service(
    reranker: Reranker,
    db_path: str = "./qdrant_db",
    collection_name: str = "docs",
) -> RAGService:
    cache_dir = os.path.join(str(Path.home()), ".cache", "fastembed")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=cache_dir)
    dense_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir=cache_dir)

    client = QdrantClient(path=db_path)

    embedder = DocumentEmbedder(
        dense_embedding_model=dense_model,
        sparse_embedding_model=sparse_model,
        batch_size=64,
        max_retries=5,
    )

    manager = QDrantManager(
        client=client,
        collection_name=collection_name,
        dense_dimension=384,
        distance=models.Distance.COSINE,
    )

    manager.create_collection()

    retriever = Retriever(
        embedder=embedder,
        vectorstore=manager,
    )

    rag = RAGService(
        reranker=reranker,
        retriever=retriever,
        qdrant_manager=manager,
        embedder=embedder,
        chunker=DocumentChunker(),
        loader=DocumentLoader(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(system_prompt=SYSTEM_PROMPT),
    )

    return rag
