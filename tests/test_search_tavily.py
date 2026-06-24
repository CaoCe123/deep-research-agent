from deep_research.search import tavily


def test_tavily_maps_results_to_unified_format(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def search(self, query, max_results=3):
            return {"results": [{"title": "T", "url": "http://x", "content": "body"}]}
    monkeypatch.setattr(tavily, "TavilyClient", FakeClient)

    out = tavily.search("q")
    assert len(out) == 1
    f = out[0]
    assert f["query"] == "q" and f["title"] == "T"
    assert f["url"] == "http://x" and f["content"] == "body"
    # academic fields present but empty
    assert f["authors"] == [] and f["doi"] is None and f["cited_by"] is None


def test_tavily_returns_empty_on_error(monkeypatch):
    class BoomClient:
        def __init__(self, *a, **k): pass
        def search(self, *a, **k): raise RuntimeError("network down")
    monkeypatch.setattr(tavily, "TavilyClient", BoomClient)

    assert tavily.search("q") == []
