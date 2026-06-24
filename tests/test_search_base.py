import pytest

from deep_research.search.base import make_finding


def test_make_finding_has_all_fields_with_defaults():
    f = make_finding(query="q", title="t", url="u", content="c")
    assert f == {
        "query": "q", "title": "t", "url": "u", "content": "c",
        "authors": [], "year": None, "doi": None, "venue": None, "cited_by": None,
    }


def test_make_finding_populates_academic_fields():
    f = make_finding(query="q", title="t", url="u", content="c",
                     authors=["A", "B"], year=2024, doi="10.1/x",
                     venue="IEEE Trans", cited_by=42)
    assert f["authors"] == ["A", "B"]
    assert f["year"] == 2024 and f["doi"] == "10.1/x"
    assert f["venue"] == "IEEE Trans" and f["cited_by"] == 42


def test_get_search_fn_known_sources_are_callable():
    from deep_research.search import get_search_fn, SOURCES
    assert set(SOURCES) == {"tavily", "openalex"}
    assert callable(get_search_fn("tavily"))
    assert callable(get_search_fn("openalex"))


def test_get_search_fn_unknown_source_raises():
    from deep_research.search import get_search_fn
    with pytest.raises(ValueError):
        get_search_fn("scopus")
