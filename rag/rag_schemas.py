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
    

