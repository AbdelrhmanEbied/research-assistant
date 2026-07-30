from sentence_transformers import CrossEncoder
from schemas import RetrievedDocuments
from collections.abc import Sequence

class reranker:
    def __init__(
            self,
            model: str, 
            device: str 
    ):
        self.model = CrossEncoder(
            model,
            device = device,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocuments],
        top_k: int = 5,
    ) -> list[RetrievedDocuments]:
        
        if not documents:
            return []
        
        topk = min(top_k,len(documents))
        
        pairs = [
            (query,doc.text)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents,scores),
            key = lambda x:x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:topk]]

    