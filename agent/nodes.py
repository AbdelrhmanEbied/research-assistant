from agent.agent_schemas import AgentState, KnowledgeSource, PromptMode, RouteQuery
from agent.llms import (
    extract_llm_text,
    extract_usage_tokens,
    get_llms,
    get_request_api_key,
)
from agent.prompts import CLASSIFIER_SYSTEM_PROMPT
from rag.rag_schemas import SearchType, build_sources
from settings import get_settings_store
from telemetry import get_current_tracker
from telemetry.tokens import estimate_token_counts

#: Explicit user overrides for the classified mode/source. ``None`` means the
#: classifier decides as usual (the "auto" behaviour).
SOURCE_OVERRIDES: dict[str, KnowledgeSource] = {
    "documents": KnowledgeSource.RAG,
    "web": KnowledgeSource.WEB,
    "chat": KnowledgeSource.NONE,
}
MODE_OVERRIDES: dict[str, PromptMode] = {mode.value: mode for mode in PromptMode}


def _llm_config(state: AgentState) -> dict:
    return state.get("llm_config") or {}


def _request_llms(state: AgentState):
    cfg = _llm_config(state)
    return get_llms(
        model=cfg.get("model"),
        model_provider=cfg.get("model_provider"),
        api_key=get_request_api_key(),
    )


def classify_request(state: AgentState):
    """
    Analyze the user's query and determine the execution mode and
    required knowledge source.

    When the user provided an explicit ``mode``/``source`` override (search
    mode selector, compare, summarize, ...) it wins; otherwise the classifier
    LLM routes the request to one of the supported prompt modes (chat,
    summarize, compare, or explain) and decides whether the response should
    rely on the LLM alone, the RAG pipeline, or live web search.
    """
    query = state["query"]
    mode_override = state.get("mode_override")
    source_override = state.get("source_override")

    forced_source = SOURCE_OVERRIDES.get(source_override)
    forced_mode = MODE_OVERRIDES.get(mode_override)

    # "chat" is a full override: answer from the LLM alone, no routing needed.
    if source_override == "chat":
        forced_source = KnowledgeSource.NONE
        forced_mode = PromptMode.CHAT

    mode = None
    source = None

    if forced_source is None or forced_mode is None:
        _, classifier_llm = _request_llms(state)

        tracker = get_current_tracker()

        with tracker.span("classify_request", span_type="CLASSIFIER"):
            result: RouteQuery = classifier_llm.invoke([
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ])
        mode = mode or result.mode
        source = source or result.source

    mode = forced_mode or mode or PromptMode.CHAT
    source = forced_source or source or KnowledgeSource.NONE

    tracker = get_current_tracker()
    tracker.add_tag("source", source.value)
    tracker.add_tag("mode", mode.value)

    return {
        "mode": mode,
        "source": source,
    }


def make_prepare_prompt_node(rag, search_service):
    def prepare_prompt(state: AgentState):
        """
        Prepare the prompt and supporting knowledge for response generation.

        Selects the appropriate knowledge pipeline based on the classified
        knowledge source:
        - WEB: Retrieve and prepare live web search results.
        - RAG: Retrieve relevant documents from the vector database.
        - NONE: Build a prompt without external retrieval.

        Returns a populated `KnowledgeResult` containing the retrieved
        documents (if any), constructed context, and final prompt, plus a
        flat, JSON-safe list of sources for the frontend citations.
        """

        tracker = get_current_tracker()

        with tracker.span("prepare_prompt", span_type="AGENT"):
            source = state["source"]
            if isinstance(source, str):
                source = KnowledgeSource(source)

            mode = state["mode"]
            if isinstance(mode, str):
                mode = PromptMode(mode)

            history = state.get("history", [])

            retrieval_config = _resolve_retrieval_config(
                state.get("retrieval_config") or {}
            )

            if source == KnowledgeSource.WEB:
                knowledge_result = search_service.search(
                    query=state["query"],
                    mode=mode,
                    history=history,
                    max_results=retrieval_config["limit"],
                    search_depth=retrieval_config["search_depth"],
                )
            elif source == KnowledgeSource.RAG:
                knowledge_result = rag.prepare(
                    query=state["query"],
                    mode=mode,
                    retrieve=True,
                    history=history,
                    conversation_id=state.get("conversation_id"),
                    search_type=retrieval_config["search_type"],
                    limit=retrieval_config["limit"],
                    rerank=retrieval_config["rerank"],
                    rerank_top_k=retrieval_config["rerank_top_k"],
                )
            else:
                knowledge_result = rag.prepare(
                    query=state["query"],
                    mode=mode,
                    retrieve=False,
                    history=history,
                    conversation_id=state.get("conversation_id"),
                )

        return {
            "knowledge_result": knowledge_result,
            "sources": build_sources(knowledge_result),
        }

    return prepare_prompt


def _resolve_retrieval_config(overrides: dict) -> dict:
    """Merge the user's per-request retrieval overrides onto the persisted
    defaults so the backend always has a complete, valid set of options."""
    defaults = get_settings_store().get_retrieval()
    search_type = overrides.get("search_type") or defaults["search_type"]
    if search_type not in {st.value for st in SearchType}:
        search_type = "hybrid"

    limit = overrides.get("limit")
    if limit is None:
        limit = defaults["limit"]
    limit = max(1, min(int(limit), 50))

    rerank = overrides.get("rerank")
    rerank = defaults["rerank"] if rerank is None else bool(rerank)

    rerank_top_k = overrides.get("rerank_top_k") or defaults["rerank_top_k"]
    rerank_top_k = max(1, min(int(rerank_top_k), limit))

    search_depth = overrides.get("search_depth") or defaults["search_depth"]
    if search_depth not in {"basic", "advanced"}:
        search_depth = "basic"

    tracker = get_current_tracker()
    tracker.add_tag("search_type", search_type)
    tracker.add_tag("retrieval_limit", str(limit))
    tracker.add_tag("rerank", "true" if rerank else "false")
    tracker.add_tag("search_depth", search_depth)

    return {
        "search_type": search_type,
        "limit": limit,
        "rerank": rerank,
        "rerank_top_k": rerank_top_k,
        "search_depth": search_depth,
    }


def generate_answer(state: AgentState):
    tracker = get_current_tracker()

    prompt = state["knowledge_result"].prompt
    llm, _ = _request_llms(state)

    with tracker.span("generate_answer", span_type="LLM", latency_metric="llm_latency_ms"):
        response = llm.invoke(prompt)

    text = extract_llm_text(response)

    tokens = extract_usage_tokens(response)
    if tokens is not None:
        tracker.add_metric("input_tokens", tokens["input_tokens"])
        tracker.add_metric("output_tokens", tokens["output_tokens"])
        tracker.add_metric("total_tokens", tokens["total_tokens"])
    else:
        estimated = estimate_token_counts(prompt, text)
        if estimated is not None:
            tracker.add_metric("input_tokens", estimated[0])
            tracker.add_metric("output_tokens", estimated[1])
            tracker.add_metric("total_tokens", estimated[2])

    return {
        "response": text,
    }
