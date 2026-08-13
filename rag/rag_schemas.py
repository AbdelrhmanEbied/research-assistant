from dataclasses import dataclass
from enum import Enum
from typing import Any

from qdrant_client import models


@dataclass
class EmbeddedDocument:
    id: str
    text: str
    dense: list[float]
    sparse: models.SparseVector
    metadata: dict[str, Any]


@dataclass
class EmbeddedQuery:
    dense: list[float]
    sparse: models.SparseVector



class SearchType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

@dataclass
class RetrievedDocuments:
    text: str
    metadata: dict
    score: float

@dataclass 

class Context:
    text: str
    sources: list[dict]


@dataclass(slots=True)
class KnowledgeResult:
    query: str
    retrieved_documents: list[RetrievedDocuments]
    reranked_documents: list[RetrievedDocuments]
    context: Context
    prompt: str


def _domain_of(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlparse

    host = urlparse(url).netloc or None
    if host and host.startswith("www."):
        host = host[4:]
    return host


def _snippet_of(text: str, limit: int = 220) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_sources(result: KnowledgeResult) -> list[dict]:
    """Flatten the reranked documents into plain, JSON-safe citation dicts.

    RAG sources carry a document ``name``/``document_id`` plus page/chunk
    information; web sources carry a ``title``/``url`` plus domain and a short
    snippet. Only documents actually retrieved and used are included. Internal
    payload keys are never exposed.
    """
    sources = []
    for doc in result.reranked_documents:
        metadata = doc.metadata or {}
        url = metadata.get("url")
        is_web = bool(url)

        label = metadata.get("name") or metadata.get("title") or "Source"

        entry = {
            "source": "web" if is_web else "rag",
            "label": label,
            "name": metadata.get("name"),
            "title": metadata.get("title"),
            "url": url,
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "page": metadata.get("page"),
        }

        if is_web:
            entry["domain"] = _domain_of(url)
            entry["snippet"] = _snippet_of(doc.text)
        else:
            entry["snippet"] = _snippet_of(doc.text)

        sources.append(entry)
    return sources
    

