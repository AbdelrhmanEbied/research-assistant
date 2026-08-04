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
        return self.splitter.split_documents(documents)
