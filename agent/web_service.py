import os

from dotenv import load_dotenv
from tavily import TavilyClient

from agent.agent_schemas import ChatMessage, PromptMode
from agent.prompts import SYSTEM_PROMPT
from rag.builders import ContextBuilder, PromptBuilder
from rag.rag_schemas import KnowledgeResult, RetrievedDocuments
from rag.reranker import Reranker
from telemetry import get_current_tracker

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

#: Tavily search depths. ``basic`` is faster/cheaper, ``advanced`` is more
#: thorough and takes longer.
SEARCH_DEPTHS = ("basic", "advanced")
DEFAULT_SEARCH_DEPTH = "basic"


class WebSearchService:
    def __init__(
        self,
        tavily_client: TavilyClient,
        reranker: Reranker | None,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
    ):
        self.client = tavily_client
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder

    def _search_tavily(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = DEFAULT_SEARCH_DEPTH,
    ) -> list[RetrievedDocuments]:
        response = self.client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
        return self._convert(response["results"])

    def _convert(self, results: list[dict]) -> list[RetrievedDocuments]:
        documents = []

        for result in results:
            documents.append(
                RetrievedDocuments(
                    text=result.get("raw_content") or result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata={
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "source": "web",
                    },
                )
            )
        return documents

    def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocuments],
    ) -> list[RetrievedDocuments]:
        if self.reranker is None:
            return documents
        else:
            return self.reranker.rerank(documents=documents, query=query)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        mode: PromptMode | str,
        history: list[ChatMessage],
        search_depth: str = DEFAULT_SEARCH_DEPTH,
    ) -> KnowledgeResult:
        tracker = get_current_tracker()

        if search_depth not in SEARCH_DEPTHS:
            search_depth = DEFAULT_SEARCH_DEPTH

        with tracker.span(
            "web_search",
            span_type="WEB",
            latency_metric="web_search_latency_ms",
        ):
            retrieved_docs = self._search_tavily(query, max_results, search_depth)

            reranked_docs = self._rerank(query, retrieved_docs)

            context = self.context_builder.build(documents=reranked_docs)

            prompt = self.prompt_builder.build(
                mode=mode,
                question=query,
                context=context,
                history=history,
            )

        tracker.add_metric("retrieved_documents", len(retrieved_docs))
        tracker.add_metric("reranked_documents", len(reranked_docs))
        tracker.add_tag("search_depth", search_depth)

        return KnowledgeResult(
            query=query,
            retrieved_documents=retrieved_docs,
            reranked_documents=reranked_docs,
            context=context,
            prompt=prompt,
        )


def create_web_search_service(reranker: Reranker) -> WebSearchService:
    return WebSearchService(
        tavily_client=tavily_client,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(system_prompt=SYSTEM_PROMPT),
        reranker=reranker,
    )
