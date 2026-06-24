from langgraph.graph import StateGraph, START, END
from deep_research.state import ResearchState


def test_findings_reducer_accumulates():
    """Two nodes each writing findings should accumulate, not overwrite."""
    def a(state): return {"findings": [{"title": "A"}]}
    def b(state): return {"findings": [{"title": "B"}]}

    g = StateGraph(ResearchState)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile()

    out = app.invoke({"topic": "t", "max_iterations": 1, "iterations": 0,
                      "sub_questions": [], "findings": [], "reflection": "", "report": ""})
    titles = [f["title"] for f in out["findings"]]
    assert titles == ["A", "B"]


def test_plan_and_reflection_schemas_exist():
    from deep_research.state import Plan, Reflection
    p = Plan(sub_questions=["q1", "q2"])
    r = Reflection(is_sufficient=False, next_query="more")
    assert p.sub_questions == ["q1", "q2"]
    assert r.is_sufficient is False and r.next_query == "more"
