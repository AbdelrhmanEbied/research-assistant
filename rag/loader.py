from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from pathlib import Path
from functools import partial
from langchain_core.documents import Document
LOADERS = {
    ".pdf": PyMuPDFLoader,      
    ".txt": partial(TextLoader, encoding="utf-8"),
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".md": UnstructuredMarkdownLoader
}


class DocumentLoader:
    def __init__(self):
        self.loaders = LOADERS.copy()

    def _load_file(self,path:Path) -> list[Document]:
        suffix = path.suffix.lower()
        loader_cls = self.loaders.get(suffix)

        if loader_cls is None:
            supported = ", ".join(sorted(self.loaders))
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {supported}"
            )
        loader = loader_cls(str(path))

        return loader.load()
    
    def _load_directory(self, path: Path) -> list[Document]:
        documents = []

        for ext in self.loaders:
            loader = DirectoryLoader(
                str(path),
                glob=f"**/*{ext}",
                loader_cls=self.loaders[ext],
                recursive=True,
                show_progress=True,
                use_multithreading=True,
            )

            documents.extend(loader.load())

        return documents

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            return self._load_directory(path)

        return self._load_file(path)


loader = DocumentLoader()
documents = loader.load("./langgraph/quickstart.txt")
print(documents[0].metadata)