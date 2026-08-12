from agent.agent_schemas import AgentState, KnowledgeSource, PromptMode, RouteQuery
from agent.llms import classifier_llm,llm
from agent.prompts import CLASSIFIER_SYSTEM_PROMPT
from telemetry import get_current_tracker
from telemetry.tokens import estimate_token_counts


def classify_request(state: AgentState):
    """
    Analyze the user's query and determine the execution mode and
    required knowledge source.

    Uses the classifier LLM to route the request to one of the supported
    prompt modes (chat, summarize, compare, or explain) and decide
    whether the response should rely on the LLM alone, the RAG pipeline,
    or live web search.
    """
    query = state["query"]

    tracker = get_current_tracker()

    with tracker.span("classify_request", span_type="CLASSIFIER"):
        result: RouteQuery = classifier_llm.invoke([
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ])


    return {
        "mode": result.mode,
        "source": result.source,
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
        documents (if any), constructed context, and final prompt.
        """

        tracker = get_current_tracker()

        with tracker.span("prepare_prompt", span_type="AGENT"):
            source = state["source"]
            if isinstance(source, str):
                source = KnowledgeSource(source)

            mode = state["mode"]
            if isinstance(mode, str):
                mode = PromptMode(mode)

            history = state["history"]

            if source == KnowledgeSource.WEB:
                knowledge_result = search_service.search(
                    query=state["query"],
                    mode=mode,
                    history=history,
                )
            elif source == KnowledgeSource.RAG:
                knowledge_result = rag.prepare(
                    query=state["query"],
                    mode=mode,
                    retrieve=True,
                    history=history,
                    conversation_id=state.get("conversation_id"),
                )
            else:
                knowledge_result = rag.prepare(
                    query=state["query"],
                    mode=mode,
                    retrieve=False,
                    history=history,
                    conversation_id=state.get("conversation_id"),
                )

        return {"knowledge_result": knowledge_result}
    return prepare_prompt

def generate_answer(state: AgentState):
    tracker = get_current_tracker()

    prompt = state["knowledge_result"].prompt

    with tracker.span("generate_answer", span_type="LLM", latency_metric="llm_latency_ms"):
        response = llm.invoke(prompt)

    text = response.content[0]["text"]

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is not None and output_tokens is not None:
        tracker.add_metric("input_tokens", input_tokens)
        tracker.add_metric("output_tokens", output_tokens)
        tracker.add_metric("total_tokens", usage.get("total_tokens") or input_tokens + output_tokens)
    else:
        estimated = estimate_token_counts(prompt, text)
        if estimated is not None:
            tracker.add_metric("input_tokens", estimated[0])
            tracker.add_metric("output_tokens", estimated[1])
            tracker.add_metric("total_tokens", estimated[2])

    return {
        "response": text,
        "history": [
            {
                "role": "assistant",
                "content": text,
            }
        ],
    }



