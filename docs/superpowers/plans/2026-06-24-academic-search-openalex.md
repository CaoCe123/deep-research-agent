# OpenAlex Academic Search + Literature Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a switchable OpenAlex academic-paper search source (finds IEEE papers' metadata + abstracts) to the existing Deep Research Agent, and upgrade the report into a structured literature review with a code-generated reference list.

**Architecture:** Introduce a `deep_research/search/` package with a unified result dict (`make_finding`) and a `get_search_fn(source)` dispatcher. Migrate Tavily into `search/tavily.py`, add `search/openalex.py` (stdlib `urllib`, restores OpenAlex inverted-index abstracts, sorts by citations). `search_node` dispatches on a new `search_source` state field; `write_node` deterministically appends a reference list. CLI gains `--source tavily|openalex`.

**Tech Stack:** Python 3.10, LangGraph, langchain-anthropic, stdlib urllib (no new deps), pytest. Run python via `.venv/bin/python`; run pytest with the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix (this machine's system ROS packages crash pytest autoload — environment quirk, do NOT add a workaround file to the repo).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deep_research/search/__init__.py` | `get_search_fn(source)` dispatcher + `SOURCES` registry | Create |
| `deep_research/search/base.py` | `make_finding(...)` unified result dict | Create |
| `deep_research/search/tavily.py` | Tavily search → unified format | Create (migrated) |
| `deep_research/search/openalex.py` | OpenAlex search + inverted-index abstract restore | Create |
| `deep_research/tools.py` | Keep only `slugify` (remove `tavily_search`) | Modify |
| `deep_research/state.py` | Add `search_source: str` field | Modify |
| `deep_research/nodes.py` | `search_node` dispatches via `get_search_fn`; `write_node` appends reference list | Modify |
| `main.py` | Add `--source` arg; pass `search_source` into inputs | Modify |
| `tests/test_tools.py` | Keep slugify tests only (remove tavily tests) | Modify |
| `tests/test_search_base.py` | Tests for make_finding + dispatcher | Create |
| `tests/test_search_tavily.py` | Tavily tests (migrated) | Create |
| `tests/test_search_openalex.py` | OpenAlex tests (mock urlopen) | Create |
| `tests/test_nodes.py` | Update search_node tests; add reference-list tests | Modify |
| `tests/test_cli.py` | Add `--source` tests | Modify |

**Verified OpenAlex response shape (used in fixtures below):** each work has `title`, `doi` (full URL like `https://doi.org/...`), `publication_year`, `cited_by_count`, `abstract_inverted_index` (dict word→[positions]), `authorships[].author.display_name`, `primary_location.source.display_name` (venue) and `primary_location.landing_page_url`.

---

## Task 1: Unified result format + dispatcher (`search/base.py`, `search/__init__.py`)

**Files:**
- Create: `deep_research/search/__init__.py`
- Create: `deep_research/search/base.py`
- Test: `tests/test_search_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_search_base.py`:
```python
import pytest

from deep_research.search.base import make_finding
from deep_research.search import get_search_fn, SOURCES


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
    assert set(SOURCES) == {"tavily", "openalex"}
    assert callable(get_search_fn("tavily"))
    assert callable(get_search_fn("openalex"))


def test_get_search_fn_unknown_source_raises():
    with pytest.raises(ValueError):
        get_search_fn("scopus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.search'`.

- [ ] **Step 3: Write `search/base.py`**

```python
def make_finding(query, title, url, content,
                 authors=None, year=None, doi=None, venue=None, cited_by=None) -> dict:
    """Unified search result. Academic fields stay empty/None for web sources."""
    return {
        "query": query,
        "title": title,
        "url": url,
        "content": content,
        "authors": authors or [],
        "year": year,
        "doi": doi,
        "venue": venue,
        "cited_by": cited_by,
    }
```

- [ ] **Step 4: Write `search/__init__.py`**

```python
from . import openalex, tavily

SOURCES = {
    "tavily": tavily.search,
    "openalex": openalex.search,
}


def get_search_fn(source: str):
    """Return the search() function for a source name, or raise ValueError."""
    if source not in SOURCES:
        raise ValueError(f"未知检索源: {source}；可选: {', '.join(SOURCES)}")
    return SOURCES[source]
```

Note: this imports `tavily` and `openalex` submodules, which are created in Tasks 2 and 3. Until those exist, the import fails — that's expected; this task's tests will pass only after Tasks 2 & 3. To keep TDD green per-task, **do Step 5 commit after Tasks 2 & 3 are in place**. To avoid that ordering problem, implement the submodules as part of completing this task is NOT allowed (one responsibility per task). Instead: create `search/__init__.py` with lazy imports so base-level tests pass independently:

```python
SOURCES = {"tavily": None, "openalex": None}  # populated lazily


def _load():
    from . import tavily, openalex
    SOURCES["tavily"] = tavily.search
    SOURCES["openalex"] = openalex.search


def get_search_fn(source: str):
    if SOURCES.get(source) is None:
        _load()
    if source not in SOURCES:
        raise ValueError(f"未知检索源: {source}；可选: {', '.join(SOURCES)}")
    return SOURCES[source]
```

Wait — this still imports tavily/openalex at `get_search_fn` time, and `test_get_search_fn_known_sources_are_callable` calls it. So Tasks 2 & 3 must exist first. **Resolution: reorder — implement Task 2 (tavily) and Task 3 (openalex) BEFORE this task's `__init__.py` dispatcher tests run.** See "Execution order" note below. For THIS task, write `base.py` + its 2 make_finding tests and commit; defer the dispatcher (`__init__.py` + its 2 tests) to a later step after Tasks 2 & 3.

**Execution order (revised):** Task 1a = `base.py` + make_finding tests. Task 2 = tavily. Task 3 = openalex. Task 3b = dispatcher `__init__.py` + dispatcher tests. The plan below is structured this way.

- [ ] **Step 5 (Task 1a only): Create `search/__init__.py` as an empty package marker**

`deep_research/search/__init__.py`:
```python
```
(empty for now; the dispatcher is added in Task 4.)

- [ ] **Step 6: Run the make_finding tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_base.py -k make_finding -v`
Expected: PASS (2 passed). The 2 dispatcher tests will error (no dispatcher yet) — that is expected and resolved in Task 4.

- [ ] **Step 7: Commit**

```bash
git add deep_research/search/__init__.py deep_research/search/base.py tests/test_search_base.py
git commit -m "feat: unified search result format (make_finding)"
```

---

## Task 2: Migrate Tavily into `search/tavily.py`

**Files:**
- Create: `deep_research/search/tavily.py`
- Create: `tests/test_search_tavily.py`
- Modify: `deep_research/tools.py` (remove `tavily_search`, keep `slugify`)
- Modify: `tests/test_tools.py` (remove tavily tests, keep slugify tests)

- [ ] **Step 1: Write the failing tests**

`tests/test_search_tavily.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_tavily.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.search.tavily'`.

- [ ] **Step 3: Write `search/tavily.py`**

```python
from tavily import TavilyClient

from .base import make_finding


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search Tavily; on any failure log a warning and return an empty list."""
    try:
        client = TavilyClient()
        resp = client.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] 检索失败 query={query!r}: {e}")
        return []
    return [
        make_finding(query=query, title=r.get("title", ""),
                     url=r.get("url", ""), content=r.get("content", ""))
        for r in resp.get("results", [])
    ]
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_tavily.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Remove `tavily_search` from `tools.py`**

Replace the entire contents of `deep_research/tools.py` with (slugify only):
```python
import re


def slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug, preserving CJK and alphanumerics."""
    cleaned = re.sub(r"[^\w一-鿿]+", "-", text, flags=re.UNICODE).strip("-")
    return cleaned[:max_len] or "report"
```

- [ ] **Step 6: Remove the tavily tests from `tests/test_tools.py`**

Replace the entire contents of `tests/test_tools.py` with (slugify tests only):
```python
from deep_research import tools


def test_slugify_basic():
    assert tools.slugify("Hello World") == "Hello-World"


def test_slugify_keeps_cjk_and_strips_punctuation():
    out = tools.slugify("2026 年小型语言模型（SLM）")
    assert out == "2026-年小型语言模型-SLM"


def test_slugify_truncates_and_has_fallback():
    assert len(tools.slugify("a" * 200)) == 60
    assert tools.slugify("!!!") == "report"
```

- [ ] **Step 7: Run tools + tavily tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tools.py tests/test_search_tavily.py -v`
Expected: PASS (3 slugify + 2 tavily = 5 passed). NOTE: `tests/test_nodes.py` will now FAIL because `search_node` still imports `tools.tavily_search` — that is fixed in Task 5. Do not run the full suite yet.

- [ ] **Step 8: Commit**

```bash
git add deep_research/search/tavily.py deep_research/tools.py tests/test_search_tavily.py tests/test_tools.py
git commit -m "refactor: migrate tavily_search into search/tavily.py (unified format)"
```

---

## Task 3: OpenAlex search (`search/openalex.py`)

**Files:**
- Create: `deep_research/search/openalex.py`
- Test: `tests/test_search_openalex.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_search_openalex.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_openalex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.search.openalex'`.

- [ ] **Step 3: Write `search/openalex.py`**

```python
import json
import urllib.parse
import urllib.request

from .base import make_finding

OPENALEX_URL = "https://api.openalex.org/works"


def _restore_abstract(inverted_index) -> str:
    """OpenAlex returns abstracts as a word->[positions] inverted index. Restore it."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search OpenAlex (no API key), sorted by citation count desc."""
    params = {
        "search": query,
        "sort": "cited_by_count:desc",
        "per-page": max_results,
        "mailto": "deep-research-agent@example.com",
    }
    url = f"{OPENALEX_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] OpenAlex 检索失败 query={query!r}: {e}")
        return []

    findings = []
    for w in data.get("results", []):
        authors = [a["author"]["display_name"]
                   for a in w.get("authorships", []) if a.get("author")]
        primary = w.get("primary_location") or {}
        venue = (primary.get("source") or {}).get("display_name")
        doi = w.get("doi")
        findings.append(make_finding(
            query=query,
            title=w.get("title") or "",
            url=doi or primary.get("landing_page_url") or "",
            content=_restore_abstract(w.get("abstract_inverted_index")),
            authors=authors,
            year=w.get("publication_year"),
            doi=doi,
            venue=venue,
            cited_by=w.get("cited_by_count"),
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_openalex.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add deep_research/search/openalex.py tests/test_search_openalex.py
git commit -m "feat: OpenAlex academic search with inverted-index abstract restore"
```

---

## Task 4: Search dispatcher (`search/__init__.py`)

**Files:**
- Modify: `deep_research/search/__init__.py`
- Test: `tests/test_search_base.py` (dispatcher tests already written in Task 1)

- [ ] **Step 1: Confirm the dispatcher tests currently error**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_base.py -k get_search_fn -v`
Expected: FAIL/ERROR (`get_search_fn` not importable from `deep_research.search`).

- [ ] **Step 2: Implement the dispatcher**

Replace the (empty) `deep_research/search/__init__.py` with:
```python
from . import openalex, tavily

SOURCES = {
    "tavily": tavily.search,
    "openalex": openalex.search,
}


def get_search_fn(source: str):
    """Return the search() function for a source name, or raise ValueError."""
    if source not in SOURCES:
        raise ValueError(f"未知检索源: {source}；可选: {', '.join(SOURCES)}")
    return SOURCES[source]
```

- [ ] **Step 3: Run the full search-base test file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_search_base.py -v`
Expected: PASS (4 passed — 2 make_finding + 2 dispatcher).

- [ ] **Step 4: Commit**

```bash
git add deep_research/search/__init__.py
git commit -m "feat: get_search_fn dispatcher over tavily/openalex"
```

---

## Task 5: State + search_node dispatch

**Files:**
- Modify: `deep_research/state.py`
- Modify: `deep_research/nodes.py` (search_node only)
- Modify: `tests/test_nodes.py` (search_node tests)

- [ ] **Step 1: Add `search_source` to `ResearchState`**

In `deep_research/state.py`, add the field to `ResearchState` (after `max_iterations`):
```python
class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]
    reflection: str
    iterations: int
    max_iterations: int
    search_source: str
    report: str
```

- [ ] **Step 2: Update the search_node tests**

In `tests/test_nodes.py`, replace the two search_node tests (`test_search_node_first_round_uses_subquestions` and `test_search_node_later_round_uses_reflection`) with versions that patch the dispatcher and set `search_source`:
```python
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
```
Note: the second test uses empty `content`, so `_summarize` is skipped (search_node only summarizes non-empty content per the spec) — this also asserts dispatch uses `state["search_source"]`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k search_node -v`
Expected: FAIL (search_node still calls `tools.tavily_search`; `tools` no longer has it → AttributeError/ImportError).

- [ ] **Step 4: Rewrite `search_node` in `nodes.py`**

Change the imports at the top of `deep_research/nodes.py`:
```python
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from . import search as search_pkg
from .state import Plan, Reflection, ResearchState
```
(Remove `tools` from the import — `nodes.py` no longer uses it.)

Replace `search_node` with:
```python
def search_node(state: ResearchState) -> dict:
    search_fn = search_pkg.get_search_fn(state["search_source"])
    queries = state["sub_questions"] if state["iterations"] == 0 else [state["reflection"]]
    new_findings: list[dict] = []
    for q in queries:
        for hit in search_fn(q):
            hit = dict(hit)
            hit["content"] = _summarize(hit["content"]) if hit["content"] else ""
            new_findings.append(hit)
    return {"findings": new_findings, "iterations": state["iterations"] + 1}
```

- [ ] **Step 5: Run search_node tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k search_node -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add deep_research/state.py deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: search_node dispatches on state.search_source"
```

---

## Task 6: Reference list + review report (`write_node`)

**Files:**
- Modify: `deep_research/nodes.py` (write_node + new `_format_references` helper)
- Modify: `tests/test_nodes.py` (replace write_node test, add reference tests)

- [ ] **Step 1: Write the failing tests**

In `tests/test_nodes.py`, replace `test_write_node_produces_report` with these tests:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k "references or write_node" -v`
Expected: FAIL (`nodes._format_references` does not exist).

- [ ] **Step 3: Add `_format_references` and rewrite `write_node`**

In `deep_research/nodes.py`, add the helper and replace `write_node`:
```python
def _format_references(findings: list[dict]) -> str:
    lines = ["## 参考文献"]
    for i, f in enumerate(findings, 1):
        if f.get("doi") or f.get("authors"):           # academic source
            names = f.get("authors") or []
            authors = ", ".join(names[:3]) + (" 等" if len(names) > 3 else "")
            parts = [p for p in [authors, f.get("title"),
                                 f.get("venue"), str(f["year"]) if f.get("year") else ""] if p]
            ref = ". ".join(parts)
            if f.get("doi"):
                ref += f". DOI:{f['doi']}"
            cite = f"（被引 {f['cited_by']} 次）" if f.get("cited_by") is not None else ""
            lines.append(f"[{i}] {ref} {cite}".rstrip())
        else:                                          # web source fallback
            lines.append(f"[{i}] {f.get('title', '')} — {f.get('url', '')}")
    return "\n".join(lines)


def write_node(state: ResearchState) -> dict:
    findings = state["findings"]
    sources = "\n".join(
        f"[{i + 1}] {f['title']}（{f.get('venue') or 'web'}, {f.get('year') or 'n.d.'}）\n{f['content'][:500]}"
        for i, f in enumerate(findings)
    )
    model = config.get_reasoning_model()
    resp = model.invoke([
        SystemMessage(content=(
            "你是学术文献综述撰写者。基于资料写一份结构化综述，包含："
            "摘要/研究背景、主要研究方向（按主题归类）、研究趋势与空白、结论。"
            "正文引用用 [n] 对应来源编号。不要自己编造参考文献，参考文献表会由系统附加。")),
        HumanMessage(content=f"综述主题：{state['topic']}\n\n资料来源：\n{sources}"),
    ])
    references = _format_references(findings)
    return {"report": f"{resp.content}\n\n{references}"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: PASS (all node tests, including plan/reflect unchanged).

- [ ] **Step 5: Commit**

```bash
git add deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: structured review + deterministic reference list in write_node"
```

---

## Task 7: CLI `--source` flag

**Files:**
- Modify: `main.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, add these tests (and update the existing fake-graph helper to pass `search_source` through — see Step 3):
```python
def test_parse_args_source_defaults_to_tavily():
    args = main.parse_args(["t"])
    assert args.source == "tavily"


def test_parse_args_source_can_be_openalex():
    args = main.parse_args(["t", "--source", "openalex"])
    assert args.source == "openalex"


def test_parse_args_unknown_source_exits(capsys):
    import pytest
    with pytest.raises(SystemExit):
        main.parse_args(["t", "--source", "scopus"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_cli.py -k source -v`
Expected: FAIL (`args.source` does not exist).

- [ ] **Step 3: Add `--source` to `parse_args` and pass it into inputs**

In `main.py`, add to `parse_args` (after the `--sqlite` argument):
```python
    parser.add_argument("--source", default="tavily", choices=["tavily", "openalex"],
                        help="检索源：tavily（网页）| openalex（学术论文）")
```

In `main()`, update the `inputs` dict to include the source:
```python
    inputs = {"topic": args.topic, "max_iterations": args.max_iters,
              "search_source": args.source}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all CLI tests). The existing `test_main_writes_report_file` / `test_main_exits_nonzero_on_empty_report` use a fake graph that ignores inputs, so adding `search_source` to inputs does not break them.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: --source flag to switch tavily/openalex search"
```

---

## Task 8: Full suite, integration smoke, README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full unit suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`
Expected: all tests pass, no network. If any test references the removed `tools.tavily_search`, fix it (should already be handled by Tasks 2 & 5).

- [ ] **Step 2: Mocked end-to-end graph run (no network)**

Run this to confirm the graph wires together with `search_source` and produces a report with a reference list:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c "
from unittest.mock import patch
from deep_research.state import Plan, Reflection
from deep_research import config
import deep_research.search as sp
class Resp:
    def __init__(s,c): s.content=c
class FM:
    def __init__(s,structured=None,content=''): s.s=structured; s.c=content
    def with_structured_output(s,sch): return s
    def invoke(s,m): return s.s if s.s is not None else Resp(s.c)
seq=[FM(structured=Plan(sub_questions=['q1'])), FM(structured=Reflection(is_sufficient=True,next_query='')), FM(content='# 综述\n[1] x')]
def fake_search(q, **k):
    return [{'query':q,'title':'P','url':'https://doi.org/10.1/x','content':'abs',
             'authors':['Jane Doe'],'year':2021,'doi':'https://doi.org/10.1/x','venue':'IEEE','cited_by':10}]
with patch.object(config,'get_reasoning_model',lambda: seq.pop(0)), \
     patch.object(config,'get_summary_model',lambda: FM(content='摘要')), \
     patch.object(sp,'get_search_fn',lambda src: fake_search):
    from deep_research.graph import build_graph
    app=build_graph().compile()
    out=app.invoke({'topic':'T','max_iterations':1,'search_source':'openalex'})
    assert '## 参考文献' in out['report'] and 'Jane Doe' in out['report']
    print('E2E OK; report tail:'); print(out['report'][-200:])
"
```
Expected: prints `E2E OK` and a report ending with a `## 参考文献` block containing `Jane Doe`.

- [ ] **Step 3: Real OpenAlex smoke test (requires network, no extra key)**

Run (needs ANTHROPIC/AGIBOT + TAVILY keys present for the model calls; OpenAlex itself needs no key):
```bash
.venv/bin/python main.py "deep learning for wireless communication" --source openalex --max-iters 1
```
Expected: prints node progress and `报告已写入 reports/...md`; the report contains a `## 参考文献` section with DOI/citation counts and at least one IEEE source. If no API keys are available, skip and note it.

- [ ] **Step 4: Update README**

In `README.md`, under the CLI 参数 table add a row for `--source`:
```markdown
| `--source` | `tavily` | 检索源：`tavily`（网页）/ `openalex`（学术论文，可搜 IEEE 等） |
```
And add a short section after the table:
```markdown
### 学术检索（OpenAlex）

`--source openalex` 切换到学术论文检索，基于 [OpenAlex](https://openalex.org)（无需 API key），
可检索到 IEEE 等出版商论文的标题/作者/年份/DOI/被引数/摘要，并按被引数降序优先纳入高被引论文。
此时报告升级为结构化文献综述，末尾附**由系统确定性生成的参考文献表**（不依赖模型，杜绝引用幻觉）。

> 注：免费方案只能获取论文的元数据与摘要；IEEE 等的全文 PDF 在付费墙后，不在覆盖范围内。
```

- [ ] **Step 5: Commit and push**

```bash
git add README.md
git commit -m "docs: document --source openalex academic search"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** OpenAlex source (T3) ✅; unified format + dispatcher (T1, T4) ✅; Tavily migration (T2) ✅; state.search_source (T5) ✅; search_node dispatch (T5) ✅; review + deterministic reference list (T6) ✅; CLI --source (T7) ✅; citation-desc sort (T3, asserted in test) ✅; inverted-index restore (T3) ✅; error handling returns [] (T2, T3) ✅; unknown source rejected (T7 via argparse choices; dispatcher ValueError T1/T4) ✅; tests all mock network (T1-T7) ✅; integration + real smoke + README (T8) ✅.
- **Type consistency:** `make_finding` field names (`authors/year/doi/venue/cited_by`) used identically in tavily.py, openalex.py, `_format_references`, write_node sources string, and all test fixtures. `get_search_fn(source)` signature consistent across `search/__init__.py`, `search_node`, and tests. `search_source` state field used in state.py, search_node, main.py inputs, and tests.
- **Ordering risk resolved:** Task 1 ships only `base.py` + empty `__init__.py` (make_finding tests pass; dispatcher tests deferred). Tasks 2 & 3 add the submodules. Task 4 adds the dispatcher and turns the 2 deferred tests green. Tasks 2 Step 7 explicitly notes test_nodes.py is temporarily red until Task 5 — expected and called out.
- **No placeholders:** every code/test step contains complete code and exact commands with expected output.
