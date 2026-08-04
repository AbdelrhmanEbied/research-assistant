from qdrant_client import QdrantClient, models

from rag.rag_schemas import EmbeddedDocument, EmbeddedQuery, SearchType


class QDrantManager:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_dimension: int | None = 768,
        distance: models.Distance = models.Distance.COSINE,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_dimension = dense_dimension
        self.distance = distance

    def create_collection(self):
        if self.client.collection_exists(self.collection_name):
            return f"Collection: {self.collection_name} already exists."
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense":models.VectorParams(
                        size=self.dense_dimension,
                        distance = self.distance,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
    def upsert_documents(self,documents:list[EmbeddedDocument],batch_size : int = 100):
        BATCH_SIZE = batch_size
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]

            points = [
                models.PointStruct(
                    id=doc.id,
                    vector={
                        "dense": doc.dense,
                        "sparse": doc.sparse,
                    },
                    payload={
                        **doc.metadata,
                        "text": doc.text,
                    }
                )
                for doc in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )

    def delete_collection(self):
        self.client.delete_collection(
            collection_name=self.collection_name,
        )

    def collection_info(self):
        return self.client.get_collection(
            collection_name=self.collection_name,
        )

    def delete_document(
            self,
            document_id: str,
    ):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id,
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_chunk(
            self,
            chunk_id: str,
    ):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[chunk_id],
            ),
            wait=True,
        )

    def delete_by_filter(
        self,
        qdrant_filter: models.Filter,
    ):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=qdrant_filter,
            ),
            wait=True,
        )

    def search(
            self,
            query:EmbeddedQuery,
            search_type: SearchType,
            limit: int = 5
    ):
        match search_type:

            case SearchType.DENSE:
                return self.client.query_points(
                    collection_name  = self.collection_name,
                    using = "dense",
                    with_payload= True,
                    limit = limit,
                    query = query.dense
                )

            case SearchType.SPARSE:
                return self.client.query_points(
                    collection_name = self.collection_name,
                    using = "sparse",
                    limit = limit,
                    with_payload=True,
                    query = query.sparse,
                )

            case SearchType.HYBRID:
                return self.client.query_points(
                    collection_name = self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query = query.dense,
                            using = "dense"
                        ),
                        models.Prefetch(
                            query = query.sparse,
                            using = "sparse"
                        )
                    ],
                    limit = limit,
                    with_payload = True,
                    query = models.FusionQuery(
                        fusion=models.Fusion.RRF,
                    ),
                )

    def close_client(
            self,
    ):
        return self.client.close()
