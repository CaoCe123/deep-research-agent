# Deep Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable local CLI Deep Research Agent in LangGraph that runs `plan → search → reflect →(conditional loop)→ write` and produces a cited Markdown report.

**Architecture:** Flat `deep_research/` package (config, state, tools, nodes, graph) plus a thin `main.py` CLI. State flows through a LangGraph state machine; `reflect` drives a conditional loop back to `search` with an iteration circuit-breaker. Tavily provides web search; Opus 4.8 does reasoning, Haiku summarizes sources. SQLite checkpointer persists runs.

**Tech Stack:** Python 3.10, LangGraph, LangChain, langchain-anthropic, langgraph-checkpoint-sqlite, tavily-python, pytest.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Runtime dependencies |
| `deep_research/__init__.py` | Package marker |
| `deep_research/state.py` | `ResearchState` TypedDict + reducers; `Plan` / `Reflection` schemas |
| `deep_research/config.py` | Model name constants + `ChatAnthropic` factory functions |
| `deep_research/tools.py` | `slugify()`; `tavily_search()` with error handling |
| `deep_research/nodes.py` | `plan_node` / `search_node` / `reflect_node` / `write_node` |
| `deep_research/graph.py` | `route_after_reflect()`; `build_graph()` |
| `main.py` | CLI: `parse_args` / `check_keys` / `report_path` / `main` |
| `tests/` | Unit tests for control flow & data contracts (LLM/Tavily mocked) |

Test files mirror the package: `tests/test_state.py`, `tests/test_tools.py`, `tests/test_config.py`, `tests/test_nodes.py`, `tests/test_graph.py`, `tests/test_cli.py`.

---

## Task 1: Project setup & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `deep_research/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `requirements.txt`**

```
langgraph>=0.2
langchain>=0.3
langchain-anthropic>=0.3
langgraph-checkpoint-sqlite>=2.0
tavily-python>=0.5
```

- [ ] **Step 2: Create the package and test package markers**

`deep_research/__init__.py`:
```python
"""Deep Research Agent — a LangGraph-based research pipeline."""
```

`tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 3: Create a virtualenv and install dependencies**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt pytest
```
Expected: installs complete without error. (Network required.)

- [ ] **Step 4: Verify imports work**

Run:
```bash
.venv/bin/python -c "import langgraph, langchain_anthropic, tavily; from langgraph.checkpoint.sqlite import SqliteSaver; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Add `.venv/` to .gitignore if missing**

`.gitignore` already ignores `.venv/` (created during repo setup). Verify with:
```bash
grep -q ".venv" .gitignore && echo "already ignored"
```
Expected: `already ignored`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt deep_research/__init__.py tests/__init__.py
git commit -m "chore: project skeleton and dependencies"
```

---

## Task 2: State and schemas (`state.py`)

**Files:**
- Create: `deep_research/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test for the `findings` reducer**

`tests/test_state.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.state'`.

- [ ] **Step 3: Write `state.py`**

```python
import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]  # reducer: accumulate across rounds
    reflection: str
    iterations: int
    max_iterations: int
    report: str


class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3-5 个互补、可独立检索的子问题")


class Reflection(BaseModel):
    is_sufficient: bool = Field(description="现有资料是否足以写出高质量报告")
    next_query: str = Field(description="若不足，下一步最该补充检索的查询；足够则留空字符串")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add deep_research/state.py tests/test_state.py
git commit -m "feat: research state with accumulating findings reducer and schemas"
```

---

## Task 3: Tools — slugify & Tavily search (`tools.py`)

**Files:**
- Create: `deep_research/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.tools'`.

- [ ] **Step 3: Write `tools.py`**

```python
import re

from tavily import TavilyClient


def slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug, preserving CJK and alphanumerics."""
    cleaned = re.sub(r"[^\w一-鿿]+", "-", text, flags=re.UNICODE).strip("-")
    return cleaned[:max_len] or "report"


def tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """Search Tavily; on any failure log a warning and return an empty list."""
    try:
        client = TavilyClient()
        resp = client.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 — one failed query must not abort the round
        print(f"[warn] 检索失败 query={query!r}: {e}")
        return []
    return [
        {
            "query": query,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tools.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add deep_research/tools.py tests/test_tools.py
git commit -m "feat: slugify and resilient Tavily search wrapper"
```

---

## Task 4: Model configuration (`config.py`)

**Files:**
- Create: `deep_research/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from deep_research import config


def test_model_name_constants():
    assert config.REASONING_MODEL == "claude-opus-4-8"
    assert config.SUMMARY_MODEL == "claude-haiku-4-5-20251001"


def test_factories_return_models_with_expected_names(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reasoning = config.get_reasoning_model()
    summary = config.get_summary_model()
    assert reasoning.model == config.REASONING_MODEL
    assert summary.model == config.SUMMARY_MODEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.config'`.

- [ ] **Step 3: Write `config.py`**

```python
from langchain_anthropic import ChatAnthropic

REASONING_MODEL = "claude-opus-4-8"          # 规划 / 反思 / 写作
SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # 检索结果摘要


def get_reasoning_model() -> ChatAnthropic:
    return ChatAnthropic(model=REASONING_MODEL, temperature=0)


def get_summary_model() -> ChatAnthropic:
    return ChatAnthropic(model=SUMMARY_MODEL, temperature=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed).
Note: if `ChatAnthropic` exposes the model as `model_name` rather than `model` in the installed version, update the test assertions to read `reasoning.model_name`. Check with `.venv/bin/python -c "from langchain_anthropic import ChatAnthropic; import os; os.environ['ANTHROPIC_API_KEY']='x'; print(ChatAnthropic(model='claude-opus-4-8').model)"`.

- [ ] **Step 5: Commit**

```bash
git add deep_research/config.py tests/test_config.py
git commit -m "feat: anthropic model factories (opus reasoning + haiku summary)"
```

---

## Task 5: Nodes (`nodes.py`)

**Files:**
- Create: `deep_research/nodes.py`
- Test: `tests/test_nodes.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_nodes.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.nodes'`.

- [ ] **Step 3: Write `nodes.py`**

```python
from langchain_core.messages import HumanMessage, SystemMessage

from . import config, tools
from .state import Plan, Reflection, ResearchState


def plan_node(state: ResearchState) -> dict:
    model = config.get_reasoning_model().with_structured_output(Plan)
    plan = model.invoke([
        SystemMessage(content="你是研究规划助手。把用户的研究问题拆成 3-5 个互补、可独立检索的子问题。"),
        HumanMessage(content=state["topic"]),
    ])
    return {"sub_questions": plan.sub_questions, "iterations": 0}


def _summarize(raw: str) -> str:
    model = config.get_summary_model()
    resp = model.invoke([
        SystemMessage(content="把以下网页正文压缩成 3-5 条要点，保留关键事实与数字。"),
        HumanMessage(content=raw[:4000]),
    ])
    return resp.content


def search_node(state: ResearchState) -> dict:
    queries = state["sub_questions"] if state["iterations"] == 0 else [state["reflection"]]
    new_findings: list[dict] = []
    for q in queries:
        for hit in tools.tavily_search(q):
            hit = dict(hit)
            hit["content"] = _summarize(hit["content"])
            new_findings.append(hit)
    return {"findings": new_findings, "iterations": state["iterations"] + 1}


def reflect_node(state: ResearchState) -> dict:
    model = config.get_reasoning_model().with_structured_output(Reflection)
    digest = "\n".join(f"- {f['title']}: {f['content'][:200]}" for f in state["findings"])
    result = model.invoke([
        SystemMessage(content="评估资料是否足够回答研究问题，并指出缺口。"),
        HumanMessage(content=f"研究问题：{state['topic']}\n\n已有资料：\n{digest}"),
    ])
    return {"reflection": "" if result.is_sufficient else result.next_query}


def write_node(state: ResearchState) -> dict:
    sources = "\n".join(
        f"[{i + 1}] {f['title']} — {f['url']}\n{f['content'][:500]}"
        for i, f in enumerate(state["findings"])
    )
    model = config.get_reasoning_model()
    resp = model.invoke([
        SystemMessage(content="你是研究报告撰写者。综合资料写一份结构化、带编号引用 [1][2] 的 Markdown 报告。"),
        HumanMessage(content=f"研究问题：{state['topic']}\n\n资料来源：\n{sources}"),
    ])
    return {"report": resp.content}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: plan/search/reflect/write nodes"
```

---

## Task 6: Graph assembly & routing (`graph.py`)

**Files:**
- Create: `deep_research/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_graph.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deep_research.graph'`.

- [ ] **Step 3: Write `graph.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_graph.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add deep_research/graph.py tests/test_graph.py
git commit -m "feat: graph assembly with reflect->search loop and circuit breaker"
```

---

## Task 7: CLI (`main.py`)

**Files:**
- Create: `main.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import pytest

import main


def test_parse_args_defaults():
    args = main.parse_args(["my topic"])
    assert args.topic == "my topic"
    assert args.max_iters == 3
    assert args.out == "reports"
    assert args.thread_id is None
    assert args.sqlite == "research.sqlite"


def test_parse_args_overrides():
    args = main.parse_args(["t", "--max-iters", "1", "--out", "o", "--thread-id", "x"])
    assert args.max_iters == 1 and args.out == "o" and args.thread_id == "x"


def test_report_path_uses_slug(tmp_path):
    p = main.report_path(str(tmp_path), "Hello World")
    assert p.name == "Hello-World.md"
    assert p.parent == tmp_path


def test_check_keys_exits_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main.check_keys()
    assert exc.value.code != 0


def test_check_keys_passes_when_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("TAVILY_API_KEY", "b")
    main.check_keys()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Write `main.py`**

```python
import argparse
import os
import sys
from pathlib import Path

from deep_research.graph import build_graph
from deep_research.tools import slugify

REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "TAVILY_API_KEY")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("topic", help="研究问题")
    parser.add_argument("--max-iters", type=int, default=3, help="检索轮数上限（熔断）")
    parser.add_argument("--out", default="reports", help="报告输出目录")
    parser.add_argument("--thread-id", default=None, help="checkpointer 线程 id（默认按 topic 生成）")
    parser.add_argument("--sqlite", default="research.sqlite", help="checkpointer 落盘路径")
    return parser.parse_args(argv)


def check_keys() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        print(f"[error] 缺少环境变量: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def report_path(out_dir: str, topic: str) -> Path:
    return Path(out_dir) / f"{slugify(topic)}.md"


def main(argv=None) -> None:
    args = parse_args(argv)
    check_keys()

    thread_id = args.thread_id or slugify(args.topic)
    inputs = {"topic": args.topic, "max_iterations": args.max_iters}
    cfg = {"configurable": {"thread_id": thread_id}}

    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(args.sqlite) as saver:
        app = build_graph().compile(checkpointer=saver)
        for event in app.stream(inputs, cfg, stream_mode="updates"):
            for node in event:
                print(f"== {node} 完成 ==")
        report = app.get_state(cfg).values.get("report", "")

    if not report.strip():
        print("[error] 资料不足，未生成报告", file=sys.stderr)
        sys.exit(1)

    out_path = report_path(args.out, args.topic)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"报告已写入 {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (24 total).

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: CLI entrypoint with arg parsing, key checks, report writing"
```

---

## Task 8: README usage + real smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README "用法" section to reflect the real CLI**

Replace the "计划中的用法" section in `README.md` with:
```markdown
## 用法

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
export TAVILY_API_KEY="tvly-..."
# 可选：开启 LangSmith 追踪
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="deep-research-agent"

.venv/bin/python main.py "你的研究问题" --max-iters 3
# 报告写入 reports/<slug>.md
```

## 开发

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -v
```
```

- [ ] **Step 2: Run the real smoke test (requires API keys)**

Run:
```bash
export ANTHROPIC_API_KEY="..." TAVILY_API_KEY="..."
.venv/bin/python main.py "LangGraph 的 checkpointer 有哪些实现" --max-iters 1
```
Expected: prints `== plan 完成 ==`, `== search 完成 ==`, `== reflect 完成 ==`, `== write 完成 ==`, then `报告已写入 reports/...md`; the file exists and contains a cited Markdown report. If no keys are available, skip this step and note it.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: real CLI usage and dev instructions"
```

- [ ] **Step 4: Push to GitHub**

```bash
git push origin main
```
Expected: all commits pushed to `https://github.com/CaoCe123/deep-research-agent`.

---

## Self-Review Notes

- **Spec coverage:** state/reducers (T2), Tavily + error handling + slug (T3), mixed models (T4), four nodes incl. Haiku summarization (T5), conditional loop + circuit breaker (T6), CLI + streaming + report file + key checks + SQLite checkpointer (T7), README + LangSmith env + smoke test (T8). All spec sections mapped.
- **Type consistency:** `ResearchState` fields, `Plan.sub_questions`, `Reflection.is_sufficient/next_query`, `config.get_reasoning_model/get_summary_model`, `tools.slugify/tavily_search`, `graph.route_after_reflect/build_graph`, `main.parse_args/check_keys/report_path` used consistently across tasks.
- **Known risk:** `ChatAnthropic` attribute name (`model` vs `model_name`) — Task 4 Step 4 documents the check and fallback. SQLite checkpointer uses the context-manager API in `main.py` only (not needed for unit tests).
