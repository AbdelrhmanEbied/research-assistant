import math

from qdrant_client import models

from rag.embedder import DocumentEmbedder
from rag.qdrant_manager import QDrantManager
from rag.rag_schemas import EmbeddedQuery, RetrievedDocuments, SearchType
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

    def _with_document(
        self, qdrant_filter: models.Filter | None, document_id: str
    ) -> models.Filter:
        must = list(qdrant_filter.must or []) if qdrant_filter else []
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )
        return models.Filter(must=must)

    def _to_documents(self, points) -> list[RetrievedDocuments]:
        return [
            RetrievedDocuments(
                text=point.payload["text"], metadata=point.payload, score=point.score
            )
            for point in points
        ]

    def _retrieve_grouped(
        self,
        query: EmbeddedQuery,
        limit: int,
        search_type: SearchType,
        qdrant_filter: models.Filter | None,
    ) -> list[RetrievedDocuments]:
        """Retrieve per-document so no single document dominates the results.

        Each in-scope document is queried independently and the results are
        merge-sorted by score, truncated to ``limit``. This keeps compare
        requests fair across documents while preserving conversation scoping.
        """
        document_ids = [doc_id for doc_id, _ in self.vectorstore.list_document_ids(qdrant_filter)]

        if len(document_ids) <= 1:
            return self._to_documents(
                self.vectorstore.search(
                    query=query,
                    search_type=search_type,
                    limit=limit,
                    qdrant_filter=qdrant_filter,
                ).points
            )

        per_document = max(1, math.ceil(limit / len(document_ids)))
        points = []

        for document_id in document_ids:
            document_filter = self._with_document(qdrant_filter, document_id)
            points.extend(
                self.vectorstore.search(
                    query=query,
                    search_type=search_type,
                    limit=per_document,
                    qdrant_filter=document_filter,
                ).points
            )

        points.sort(key=lambda point: point.score, reverse=True)
        return self._to_documents(points[:limit])

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_type: SearchType = SearchType.HYBRID,
        conversation_id: str | None = None,
        group_by_document: bool = False,
    ) -> list[RetrievedDocuments]:
        with get_current_tracker().span(
            "retrieve",
            span_type="RETRIEVER",
            latency_metric="retrieval_latency_ms",
        ):
            embedded_query = self.embedder.embed_query(query=query)
            qdrant_filter = self._conversation_filter(conversation_id)

            if group_by_document:
                return self._retrieve_grouped(
                    query=embedded_query,
                    limit=limit,
                    search_type=search_type,
                    qdrant_filter=qdrant_filter,
                )

            result = self.vectorstore.search(
                query=embedded_query,
                search_type=search_type,
                limit=limit,
                qdrant_filter=qdrant_filter,
            )

            return self._to_documents(result.points)
