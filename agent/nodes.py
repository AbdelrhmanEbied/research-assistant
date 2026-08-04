from agent.agent_schemas import AgentState, KnowledgeSource, PromptMode, RouteQuery
from agent.llms import classifier_llm,llm
from agent.prompts import CLASSIFIER_SYSTEM_PROMPT


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
            )
        else:
            knowledge_result = rag.prepare(
                query=state["query"],
                mode=mode,
                retrieve=False,
                history=history,
            )

        return {"knowledge_result": knowledge_result}
    return prepare_prompt

def generate_answer(state: AgentState):
    response = llm.invoke(state["knowledge_result"].prompt)

    text = response.content[0]["text"]

    return {
        "response": text,
        "history": [
            {
                "role": "assistant",
                "content": text,
            }
        ],
    }



