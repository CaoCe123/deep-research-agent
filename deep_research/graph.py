from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import ResearchState


def route_after_reflect(state: ResearchState) -> str:
    """Conditional edge: decide whether to loop back to search or finish."""
    if state["reflection"] == "":               # reflection deemed coverage sufficient
        return "write"
    if state["iterations"] >= state["max_iterations"]:  # circuit breaker
        return "write"
    return "search"                             # loop: keep researching


def build_graph() -> StateGraph:
    """Build (but do not compile) the research state graph."""
    builder = StateGraph(ResearchState)
    builder.add_node("plan", nodes.plan_node)
    builder.add_node("search", nodes.search_node)
    builder.add_node("reflect", nodes.reflect_node)
    builder.add_node("write", nodes.write_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "reflect")
    builder.add_conditional_edges("reflect", route_after_reflect,
                                  {"search": "search", "write": "write"})
    builder.add_edge("write", END)
    return builder
