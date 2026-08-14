from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    AgentState,
    classify_request,
    finalize_answer,
    generate_answer,
    make_agent_reason_node,
    make_execute_tools_node,
    make_prepare_prompt_node,
    route_by_agent_mode,
)
from agent.tools import build_agent_tools


def build_agent_graph(rag, search_service, checkpointer):
    tools = build_agent_tools(rag)

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify_request", classify_request)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_node("agent_reason", make_agent_reason_node(tools))
    graph_builder.add_node("execute_tools", make_execute_tools_node(tools))
    graph_builder.add_node("finalize_answer", finalize_answer)
    prepare_prompt = make_prepare_prompt_node(rag, search_service)
    graph_builder.add_node("prepare_prompt", prepare_prompt)

    graph_builder.add_edge(START, "classify_request")
    graph_builder.add_edge("classify_request", "prepare_prompt")

    graph_builder.add_conditional_edges(
        "prepare_prompt",
        route_by_agent_mode,
        {
            "fast": "generate_answer",
            "thinking": "agent_reason",
        },
    )

    graph_builder.add_edge("generate_answer", END)

    graph_builder.add_conditional_edges(
        "agent_reason",
        _has_tool_calls,
        {
            "execute_tools": "execute_tools",
            "finalize_answer": "finalize_answer",
        },
    )
    graph_builder.add_edge("execute_tools", "agent_reason")
    graph_builder.add_edge("finalize_answer", END)

    return graph_builder.compile(checkpointer=checkpointer)


def _has_tool_calls(state: AgentState) -> str:
    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "tool_calls", None):
        return "execute_tools"
    return "finalize_answer"
