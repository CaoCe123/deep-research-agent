from deep_research import nodes, config
from deep_research.state import Plan, Reflection


class _Resp:
    def __init__(self, content): self.content = content


class _FakeModel:
    """Stands in for ChatAnthropic. structured_obj is returned by invoke after
    with_structured_output(); content_obj is returned by a plain invoke().
    Records the last messages list for prompt-branch assertions."""
    def __init__(self, structured_obj=None, content=""):
        self._structured = structured_obj
        self._content = content
        self.last_messages = None
    def with_structured_output(self, schema):
        return self
    def invoke(self, messages):
        self.last_messages = messages
        return self._structured if self._structured is not None else _Resp(self._content)


def test_plan_node_returns_subquestions_and_resets_iterations(monkeypatch):
    fake = _FakeModel(structured_obj=Plan(sub_questions=["a", "b", "c"]))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.plan_node({"topic": "T", "max_iterations": 3, "search_source": "tavily"})
    assert out["sub_questions"] == ["a", "b", "c"]
    assert out["iterations"] == 0


def test_plan_node_openalex_uses_keyword_prompt(monkeypatch):
    fake = _FakeModel(structured_obj=Plan(sub_questions=["k1"]))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    nodes.plan_node({"topic": "T", "max_iterations": 3, "search_source": "openalex"})
    system_text = fake.last_messages[0].content
    assert "关键词" in system_text


def test_plan_node_tavily_uses_subquestion_prompt(monkeypatch):
    fake = _FakeModel(structured_obj=Plan(sub_questions=["q1"]))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    nodes.plan_node({"topic": "T", "max_iterations": 3, "search_source": "tavily"})
    system_text = fake.last_messages[0].content
    assert "子问题" in system_text


def test_search_node_first_round_uses_subquestions(monkeypatch):
    from deep_research import search as search_pkg
    monkeypatch.setattr(search_pkg, "get_search_fn",
                        lambda src: (lambda q, **k: [{"query": q, "title": "T", "url": "u",
                                                      "content": "raw", "authors": [], "year": None,
                                                      "doi": None, "venue": None, "cited_by": None}]))
    monkeypatch.setattr(config, "get_summary_model", lambda: _FakeModel(content="点要"))

    state = {"sub_questions": ["q1", "q2"], "reflection": "", "iterations": 0,
             "findings": [], "search_source": "tavily"}
    out = nodes.search_node(state)
    assert out["iterations"] == 1
    assert len(out["findings"]) == 2
    assert out["findings"][0]["content"] == "点要"   # summarized, not raw


def test_search_node_dispatches_on_source(monkeypatch):
    seen = {}
    from deep_research import search as search_pkg
    def fake_get_search_fn(src):
        seen["src"] = src
        return lambda q, **k: [{"query": q, "title": "T", "url": "u", "content": "",
                                "authors": [], "year": None, "doi": None,
                                "venue": None, "cited_by": None}]
    monkeypatch.setattr(search_pkg, "get_search_fn", fake_get_search_fn)
    monkeypatch.setattr(config, "get_summary_model", lambda: _FakeModel(content="s"))

    state = {"sub_questions": ["q1"], "reflection": "follow-up", "iterations": 1,
             "findings": [], "search_source": "openalex"}
    nodes.search_node(state)
    assert seen["src"] == "openalex"


def test_reflect_node_sufficient_clears_reflection(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=True, next_query="ignored"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.reflect_node({"topic": "T", "search_source": "tavily",
                              "findings": [{"title": "A", "content": "c"}]})
    assert out["reflection"] == ""


def test_reflect_node_insufficient_returns_next_query(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=False, next_query="dig deeper"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.reflect_node({"topic": "T", "search_source": "tavily", "findings": []})
    assert out["reflection"] == "dig deeper"


def test_reflect_node_openalex_uses_keyword_prompt(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=False, next_query="kw"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    nodes.reflect_node({"topic": "T", "search_source": "openalex", "findings": []})
    system_text = fake.last_messages[0].content
    assert "关键词" in system_text


def test_reflect_node_tavily_uses_default_prompt(monkeypatch):
    fake = _FakeModel(structured_obj=Reflection(is_sufficient=False, next_query="q"))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    nodes.reflect_node({"topic": "T", "search_source": "tavily", "findings": []})
    system_text = fake.last_messages[0].content
    assert "评估资料是否足够回答研究问题" in system_text


def test_format_references_academic(monkeypatch):
    findings = [{"title": "A CNN paper", "url": "https://doi.org/10.1/x",
                 "authors": ["Jane Doe", "John Roe"], "year": 2021,
                 "doi": "https://doi.org/10.1/x", "venue": "IEEE Trans. Signal",
                 "cited_by": 99, "content": "c"}]
    refs = nodes._format_references(findings)
    assert refs.startswith("## 参考文献")
    assert "[1]" in refs
    assert "Jane Doe" in refs and "John Roe" in refs
    assert "IEEE Trans. Signal" in refs and "2021" in refs
    assert "DOI:https://doi.org/10.1/x" in refs
    assert "被引 99 次" in refs


def test_format_references_truncates_authors():
    findings = [{"title": "T", "url": "u",
                 "authors": ["A", "B", "C", "D", "E"], "year": 2020,
                 "doi": "10.1/y", "venue": "V", "cited_by": 0, "content": "c"}]
    refs = nodes._format_references(findings)
    assert "A, B, C 等" in refs   # first 3 + 等


def test_format_references_web_fallback():
    findings = [{"title": "Web Page", "url": "http://x", "authors": [],
                 "year": None, "doi": None, "venue": None, "cited_by": None, "content": "c"}]
    refs = nodes._format_references(findings)
    assert "[1] Web Page — http://x" in refs


def test_write_node_appends_reference_list(monkeypatch):
    monkeypatch.setattr(config, "get_reasoning_model",
                        lambda: _FakeModel(content="# 综述正文\n[1] ..."))
    findings = [{"title": "A", "url": "https://doi.org/10.1/x", "authors": ["Jane Doe"],
                 "year": 2021, "doi": "https://doi.org/10.1/x", "venue": "IEEE", "cited_by": 5,
                 "content": "c"}]
    out = nodes.write_node({"topic": "T", "findings": findings})
    assert out["report"].startswith("# 综述正文")
    assert "## 参考文献" in out["report"]
    assert "Jane Doe" in out["report"]


def test_dedupe_findings_by_doi():
    findings = [
        {"title": "Paper A v1", "doi": "10.1/x", "authors": [], "url": "u1"},
        {"title": "Paper A v2", "doi": "10.1/x", "authors": [], "url": "u2"},
        {"title": "Paper B", "doi": "10.1/y", "authors": [], "url": "u3"},
    ]
    out = nodes._dedupe_findings(findings)
    assert [f["doi"] for f in out] == ["10.1/x", "10.1/y"]
    assert out[0]["title"] == "Paper A v1"   # first occurrence kept


def test_dedupe_findings_by_title_when_no_doi():
    findings = [
        {"title": "Same Title", "doi": None, "authors": [], "url": "u1"},
        {"title": "same title", "doi": None, "authors": [], "url": "u2"},  # case-insensitive
        {"title": "Different", "doi": None, "authors": [], "url": "u3"},
    ]
    out = nodes._dedupe_findings(findings)
    assert len(out) == 2
    assert out[0]["title"] == "Same Title" and out[1]["title"] == "Different"


def test_dedupe_findings_preserves_order_and_distinct():
    findings = [
        {"title": "A", "doi": "10.1/a", "authors": [], "url": "u"},
        {"title": "B", "doi": "10.1/b", "authors": [], "url": "u"},
        {"title": "C", "doi": "10.1/c", "authors": [], "url": "u"},
    ]
    out = nodes._dedupe_findings(findings)
    assert [f["doi"] for f in out] == ["10.1/a", "10.1/b", "10.1/c"]


def test_write_node_dedupes_before_report(monkeypatch):
    monkeypatch.setattr(config, "get_reasoning_model",
                        lambda: _FakeModel(content="# 综述"))
    findings = [
        {"title": "Dup Paper", "url": "https://doi.org/10.1/x", "authors": ["A"],
         "year": 2021, "doi": "https://doi.org/10.1/x", "venue": "IEEE", "cited_by": 5,
         "content": "c"},
        {"title": "Dup Paper", "url": "https://doi.org/10.1/x", "authors": ["A"],
         "year": 2021, "doi": "https://doi.org/10.1/x", "venue": "IEEE", "cited_by": 5,
         "content": "c"},
    ]
    out = nodes.write_node({"topic": "T", "findings": findings})
    # reference list must contain [1] but not [2] (only one unique paper)
    assert "[1]" in out["report"]
    assert "[2]" not in out["report"]
