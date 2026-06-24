from deep_research import nodes, config, tools
from deep_research.state import Plan, Reflection


class _Resp:
    def __init__(self, content): self.content = content


class _FakeModel:
    """Stands in for ChatAnthropic. structured_obj is returned by invoke after
    with_structured_output(); content_obj is returned by a plain invoke()."""
    def __init__(self, structured_obj=None, content=""):
        self._structured = structured_obj
        self._content = content
    def with_structured_output(self, schema):
        return self
    def invoke(self, messages):
        return self._structured if self._structured is not None else _Resp(self._content)


def test_plan_node_returns_subquestions_and_resets_iterations(monkeypatch):
    fake = _FakeModel(structured_obj=Plan(sub_questions=["a", "b", "c"]))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.plan_node({"topic": "T", "max_iterations": 3})
    assert out["sub_questions"] == ["a", "b", "c"]
    assert out["iterations"] == 0


def test_search_node_first_round_uses_subquestions(monkeypatch):
    monkeypatch.setattr(tools, "tavily_search",
                        lambda q, **k: [{"query": q, "title": "T", "url": "u", "content": "raw"}])
    monkeypatch.setattr(config, "get_summary_model", lambda: _FakeModel(content="点要"))

    state = {"sub_questions": ["q1", "q2"], "reflection": "", "iterations": 0, "findings": []}
    out = nodes.search_node(state)
    assert out["iterations"] == 1
    assert len(out["findings"]) == 2
    assert out["findings"][0]["content"] == "点要"   # summarized, not raw


def test_search_node_later_round_uses_reflection(monkeypatch):
    seen = []
    def fake_search(q, **k):
        seen.append(q)
        return [{"query": q, "title": "T", "url": "u", "content": "raw"}]
    monkeypatch.setattr(tools, "tavily_search", fake_search)
    monkeypatch.setattr(config, "get_summary_model", lambda: _FakeModel(content="s"))

    state = {"sub_questions": ["q1"], "reflection": "follow-up", "iterations": 1, "findings": []}
    nodes.search_node(state)
    assert seen == ["follow-up"]


def test_reflect_node_sufficient_clears_reflection(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=True, next_query="ignored"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.reflect_node({"topic": "T", "findings": [{"title": "A", "content": "c"}]})
    assert out["reflection"] == ""


def test_reflect_node_insufficient_returns_next_query(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=False, next_query="dig deeper"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.reflect_node({"topic": "T", "findings": []})
    assert out["reflection"] == "dig deeper"


def test_write_node_produces_report(monkeypatch):
    monkeypatch.setattr(config, "get_reasoning_model", lambda: _FakeModel(content="# 报告\n[1] ..."))

    out = nodes.write_node({"topic": "T",
                            "findings": [{"title": "A", "url": "u", "content": "c"}]})
    assert out["report"].startswith("# 报告")
