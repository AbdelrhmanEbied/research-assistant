from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    AgentState,
    classify_request,
    generate_answer,
    make_prepare_prompt_node,
)


def build_agent_graph(rag, search_service, checkpointer):
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify_request", classify_request)
    graph_builder.add_node("generate_answer", generate_answer)
    prepare_prompt = make_prepare_prompt_node(rag, search_service)
    graph_builder.add_node("prepare_prompt", prepare_prompt)

    graph_builder.add_edge(START, "classify_request")
    graph_builder.add_edge("classify_request", "prepare_prompt")
    graph_builder.add_edge("prepare_prompt", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    return graph_builder.compile(checkpointer=checkpointer)
