import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import models
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from rag.rag_schemas import EmbeddedDocument, EmbeddedQuery

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    def __init__(
        self,
        dense_embedding_model: Embeddings,
        sparse_embedding_model: Any,
        batch_size: int = 32,
        max_retries: int = 5,
        retryable_exceptions: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ):
        self.dense_model = dense_embedding_model
        self.sparse_model = sparse_embedding_model
        self.batch_size = batch_size

        self.retryer = Retrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(
                multiplier=2,
                min=2,
                max=60,
            ),
            retry=retry_if_exception_type(retryable_exceptions),
            before_sleep=before_sleep_log(
                logger,
                logging.WARNING,
            ),
            reraise=True,
        )

    def _batch(self, items: list[str]):
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def _retry(self, func: Callable, *args, **kwargs):
        return self.retryer(func, *args, **kwargs)

    def _to_sparse_vector(self, sparse_embedding: Any) -> models.SparseVector:
        return models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )

    def _embed_batches(
        self,
        texts: list[str],
        embed_fn: Callable[[list[str]], list[Any]],
        desc: str,
    ) -> list[Any]:
        embeddings: list[Any] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch in tqdm(
            self._batch(texts),
            total=total_batches,
            desc=desc,
        ):
            vectors = self._retry(embed_fn, batch)
            embeddings.extend(vectors)

        if len(texts) != len(embeddings):
            raise RuntimeError(
                "Embedding provider returned "
                f"{len(embeddings)} embeddings "
                f"for {len(texts)} texts."
            )

        return embeddings

    def embed_documents(
        self,
        documents: list[Document],
    ) -> list[EmbeddedDocument]:
        valid_documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not valid_documents:
            return []

        texts = [doc.page_content for doc in valid_documents]

        dense_vectors = self._embed_batches(
            texts=texts,
            embed_fn=self.dense_model.embed_documents,
            desc="Dense embedding",
        )

        sparse_embeddings = self._embed_batches(
            texts=texts,
            embed_fn=lambda batch: list(self.sparse_model.embed(batch)),
            desc="Sparse embedding",
        )

        sparse_vectors = [
            self._to_sparse_vector(vec)
            for vec in sparse_embeddings
        ]

        if len(dense_vectors) != len(sparse_vectors):
            raise RuntimeError(
                "Dense and sparse embedding counts do not match: "
                f"{len(dense_vectors)} dense vs {len(sparse_vectors)} sparse."
            )

        embedded_documents: list[EmbeddedDocument] = []
        total_chunks = len(valid_documents)
        default_document_id = str(uuid4())

        for index, (doc, dense, sparse) in enumerate(
            zip(valid_documents, dense_vectors, sparse_vectors)
        ):
            metadata = {
                **doc.metadata,
                "document_id": doc.metadata.get("document_id", default_document_id),
                "chunk_id": str(doc.metadata.get("chunk_id", uuid4())),
                "chunk_index": index,
                "total_chunks": total_chunks,
                "created_at": datetime.now(UTC).isoformat(),
            }

            embedded_documents.append(
                EmbeddedDocument(
                    id=metadata["chunk_id"],
                    text=doc.page_content,
                    dense=dense,
                    sparse=sparse,
                    metadata=metadata,
                )
            )

        return embedded_documents

    def embed_query(self, query: str) -> EmbeddedQuery:
        dense = self._retry(self.dense_model.embed_query, query)

        sparse_embedding = self._retry(
            lambda q: next(self.sparse_model.query_embed(query=q)),
            query,
        )

        return EmbeddedQuery(
            dense=dense,
            sparse=self._to_sparse_vector(sparse_embedding),
        )