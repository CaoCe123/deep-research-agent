from deep_research import tools


def test_slugify_basic():
    assert tools.slugify("Hello World") == "Hello-World"


def test_slugify_keeps_cjk_and_strips_punctuation():
    out = tools.slugify("2026 年小型语言模型（SLM）")
    assert out == "2026-年小型语言模型-SLM"


def test_slugify_truncates_and_has_fallback():
    assert len(tools.slugify("a" * 200)) == 60
    assert tools.slugify("!!!") == "report"


def test_tavily_search_maps_results(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def search(self, query, max_results=3):
            return {"results": [{"title": "T", "url": "http://x", "content": "body"}]}
    monkeypatch.setattr(tools, "TavilyClient", FakeClient)

    out = tools.tavily_search("q")
    assert out == [{"query": "q", "title": "T", "url": "http://x", "content": "body"}]


def test_tavily_search_returns_empty_on_error(monkeypatch):
    class BoomClient:
        def __init__(self, *a, **k): pass
        def search(self, *a, **k): raise RuntimeError("network down")
    monkeypatch.setattr(tools, "TavilyClient", BoomClient)

    assert tools.tavily_search("q") == []
