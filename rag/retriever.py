from qdrant_client import models

from rag.embedder import DocumentEmbedder
from rag.qdrant_manager import QDrantManager
from rag.rag_schemas import RetrievedDocuments, SearchType
from telemetry import get_current_tracker


class Retriever:
    def __init__(
        self,
        embedder: DocumentEmbedder,
        vectorstore: QDrantManager,
    ):

        self.embedder = embedder
        self.vectorstore = vectorstore

    def _conversation_filter(self, conversation_id: str | None) -> models.Filter | None:
        if conversation_id is None:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(
                        value=str(conversation_id),
                    ),
                )
            ]
        )

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_type: SearchType = SearchType.HYBRID,
        conversation_id: str | None = None,
    ) -> list[RetrievedDocuments]:
        with get_current_tracker().span(
            "retrieve",
            span_type="RETRIEVER",
            latency_metric="retrieval_latency_ms",
        ):
            embedded_query = self.embedder.embed_query(query=query)

            result = self.vectorstore.search(
                query=embedded_query,
                search_type=search_type,
                limit=limit,
                qdrant_filter=self._conversation_filter(conversation_id),
            )

            documents = []

            for point in result.points:
                payload = point.payload

                documents.append(
                    RetrievedDocuments(text=payload["text"], metadata=payload, score=point.score)
                )

            return documents
