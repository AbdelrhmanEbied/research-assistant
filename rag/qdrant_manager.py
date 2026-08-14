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
                    "dense": models.VectorParams(
                        size=self.dense_dimension,
                        distance=self.distance,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def upsert_documents(self, documents: list[EmbeddedDocument], batch_size: int = 100):
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

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
                    },
                )
                for doc in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
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

    def list_document_ids(
        self,
        qdrant_filter: models.Filter | None = None,
    ) -> list[tuple[str, str | None]]:
        """Distinct ``(document_id, name)`` pairs within ``qdrant_filter`` scope.

        Walks the collection with ``client.scroll`` (no similarity needed) and
        deduplicates on ``document_id``.
        """
        document_ids: dict[str, str | None] = {}
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                document_id = point.payload.get("document_id")
                if document_id and document_id not in document_ids:
                    document_ids[document_id] = point.payload.get("name")

            if offset is None:
                break

        return list(document_ids.items())

    def get_points_by_document(
        self,
        document_id: str,
        qdrant_filter: models.Filter | None = None,
        limit: int = 100,
    ):
        """Return a document's points ordered by ``chunk_index``.

        Combines ``qdrant_filter`` (e.g. the conversation scope) with the
        ``document_id`` condition so only in-scope chunks are returned.
        """
        document_condition = models.FieldCondition(
            key="document_id",
            match=models.MatchValue(value=document_id),
        )

        must = list(qdrant_filter.must or []) if qdrant_filter else []
        must.append(document_condition)

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            order_by=models.OrderBy(key="chunk_index"),
        )

        return points

    def search(
        self,
        query: EmbeddedQuery,
        search_type: SearchType,
        limit: int = 5,
        qdrant_filter: models.Filter | None = None,
    ):
        match search_type:
            case SearchType.DENSE:
                return self.client.query_points(
                    collection_name=self.collection_name,
                    using="dense",
                    with_payload=True,
                    limit=limit,
                    query=query.dense,
                    query_filter=qdrant_filter,
                )

            case SearchType.SPARSE:
                return self.client.query_points(
                    collection_name=self.collection_name,
                    using="sparse",
                    limit=limit,
                    with_payload=True,
                    query=query.sparse,
                    query_filter=qdrant_filter,
                )

            case SearchType.HYBRID:
                return self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=query.dense,
                            using="dense",
                            filter=qdrant_filter,
                        ),
                        models.Prefetch(
                            query=query.sparse,
                            using="sparse",
                            filter=qdrant_filter,
                        ),
                    ],
                    limit=limit,
                    with_payload=True,
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF,
                    ),
                )

    def close_client(
        self,
    ):
        return self.client.close()
