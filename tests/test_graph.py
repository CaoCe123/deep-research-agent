from deep_research.graph import route_after_reflect, build_graph


def test_route_sufficient_goes_to_write():
    assert route_after_reflect({"reflection": "", "iterations": 1, "max_iterations": 3}) == "write"


def test_route_circuit_breaker_goes_to_write():
    assert route_after_reflect({"reflection": "more", "iterations": 3, "max_iterations": 3}) == "write"


def test_route_insufficient_under_limit_goes_to_search():
    assert route_after_reflect({"reflection": "more", "iterations": 1, "max_iterations": 3}) == "search"


def test_build_graph_has_all_nodes_and_compiles():
    builder = build_graph()
    app = builder.compile()  # no checkpointer needed for structure check
    nodes = set(app.get_graph().nodes)
    assert {"plan", "search", "reflect", "write"} <= nodes
