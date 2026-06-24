import json
from io import BytesIO

from deep_research.search import openalex


def test_restore_abstract_orders_words_by_position():
    inv = {"Deep": [0], "learning": [1], "works": [2]}
    assert openalex._restore_abstract(inv) == "Deep learning works"


def test_restore_abstract_handles_repeated_words():
    inv = {"the": [0, 2], "cat": [1], "sat": [3]}
    assert openalex._restore_abstract(inv) == "the cat the sat"


def test_restore_abstract_empty_returns_empty_string():
    assert openalex._restore_abstract(None) == ""
    assert openalex._restore_abstract({}) == ""


def _fake_urlopen_factory(payload):
    class _Ctx:
        def __enter__(self_):
            return BytesIO(json.dumps(payload).encode("utf-8"))
        def __exit__(self_, *a):
            return False
    def _fake_urlopen(url, timeout=None):
        _fake_urlopen.last_url = url
        return _Ctx()
    return _fake_urlopen


def test_search_maps_fields(monkeypatch):
    payload = {"results": [{
        "title": "A CNN paper",
        "doi": "https://doi.org/10.1/x",
        "publication_year": 2021,
        "cited_by_count": 99,
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [
            {"author": {"display_name": "Jane Doe"}},
            {"author": {"display_name": "John Roe"}},
        ],
        "primary_location": {"source": {"display_name": "IEEE Trans. Signal"},
                             "landing_page_url": "http://lp"},
    }]}
    monkeypatch.setattr(openalex.urllib.request, "urlopen",
                        _fake_urlopen_factory(payload))

    out = openalex.search("cnn", max_results=1)
    assert len(out) == 1
    f = out[0]
    assert f["title"] == "A CNN paper"
    assert f["content"] == "Hello world"
    assert f["authors"] == ["Jane Doe", "John Roe"]
    assert f["year"] == 2021 and f["cited_by"] == 99
    assert f["doi"] == "https://doi.org/10.1/x"
    assert f["venue"] == "IEEE Trans. Signal"
    assert f["url"] == "https://doi.org/10.1/x"


def test_search_requests_citation_sort(monkeypatch):
    monkeypatch.setattr(openalex.urllib.request, "urlopen",
                        _fake_urlopen_factory({"results": []}))
    openalex.search("anything")
    assert "sort=cited_by_count" in openalex.urllib.request.urlopen.last_url


def test_search_returns_empty_on_error(monkeypatch):
    def boom(url, timeout=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(openalex.urllib.request, "urlopen", boom)
    assert openalex.search("q") == []
