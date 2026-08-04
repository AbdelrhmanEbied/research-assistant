from rag.embedder import DocumentEmbedder
from rag.qdrant_manager import QDrantManager
from rag.rag_schemas import RetrievedDocuments, SearchType


class Retriever:
    def __init__(
            self,
            embedder: DocumentEmbedder,
            vectorstore:QDrantManager,
    ):

        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(
            self,
            query:str,
            limit: int = 5,
            search_type: SearchType = SearchType.HYBRID,
    ) -> list[RetrievedDocuments]:
        embedded_query = self.embedder.embed_query(query=query)

        result = self.vectorstore.search(
            query= embedded_query,
            search_type= search_type,
            limit = limit
        )

        documents = []

        for point in result.points:
            payload = point.payload

            documents.append(
                RetrievedDocuments(
                    text = payload['text'],
                    metadata = payload,
                    score = point.score
                )
            )

        return documents