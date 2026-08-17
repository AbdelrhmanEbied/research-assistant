import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(
        self,
        documents: list[Document],
    ) -> list[Document]:
        chunks = self.splitter.split_documents(documents)

        # Deterministic per-document chunk ids (uuid5 of ``<document_id>:<index>``)
        # so chunk-level evaluation datasets stay reproducible. Qdrant requires
        # valid UUID point ids, hence uuid5 instead of a plain string.
        counters: dict[str, int] = {}
        for chunk in chunks:
            document_id = chunk.metadata.get("document_id")
            if document_id is not None and not chunk.metadata.get("chunk_id"):
                index = counters.get(document_id, 0)
                counters[document_id] = index + 1
                chunk.metadata = {
                    **chunk.metadata,
                    "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{index}")),
                }

        return chunks
