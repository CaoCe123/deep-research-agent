# Academic Query Optimization + Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make openalex mode generate precise English keyword queries (via source-specific prompts in plan_node/reflect_node) and deduplicate findings by DOI before report generation.

**Architecture:** Three changes, all in `deep_research/nodes.py`: (1) `plan_node` selects a prompt by `state["search_source"]`; (2) `reflect_node` does the same; (3) a new `_dedupe_findings` helper runs at the top of `write_node`. No graph, state, search-source, or `make_finding` changes. `ResearchState` already carries `search_source`.

**Tech Stack:** Python 3.10, LangGraph, langchain-anthropic, pytest. Run python via `.venv/bin/python`; run pytest with the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix (this machine's system ROS packages crash pytest autoload — environment quirk, do NOT add a workaround file to the repo).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deep_research/nodes.py` | plan/reflect source-specific prompts; `_dedupe_findings`; write_node dedups | Modify |
| `tests/test_nodes.py` | `_FakeModel` records messages; new prompt-branch + dedup tests; fix existing tests missing `search_source` | Modify |

All node function signatures and the `Plan`/`Reflection` schemas are unchanged.

**Current code reference (in `deep_research/nodes.py`):**
- `plan_node` uses a single hardcoded SystemMessage "你是研究规划助手…子问题。"
- `reflect_node` uses a single hardcoded SystemMessage "评估资料是否足够回答研究问题，并指出缺口。"
- `write_node` starts with `findings = state["findings"]` (no dedup) then builds `sources`, calls model, calls `_format_references(findings)`.

**Current test helper (`tests/test_nodes.py`):** `_FakeModel.invoke(self, messages)` returns `self._structured` or `_Resp(self._content)` but does NOT record `messages`.

---

## Task 1: `_FakeModel` records messages + fix existing tests for `search_source`

This task makes no production change — it prepares the test harness so later tasks can assert prompt branching, and fixes the two existing tests that will KeyError once plan/reflect read `state["search_source"]`.

**Files:**
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Make `_FakeModel.invoke` record the messages it received**

In `tests/test_nodes.py`, replace the `_FakeModel` class with this version (adds `self.last_messages`):
```python
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
```

- [ ] **Step 2: Fix the existing plan_node test to pass `search_source`**

Replace `test_plan_node_returns_subquestions_and_resets_iterations` with:
```python
def test_plan_node_returns_subquestions_and_resets_iterations(monkeypatch):
    fake = _FakeModel(structured_obj=Plan(sub_questions=["a", "b", "c"]))
    monkeypatch.setattr(config, "get_reasoning_model", lambda: fake)

    out = nodes.plan_node({"topic": "T", "max_iterations": 3, "search_source": "tavily"})
    assert out["sub_questions"] == ["a", "b", "c"]
    assert out["iterations"] == 0
```

- [ ] **Step 3: Fix the two existing reflect_node tests to pass `search_source`**

Replace both reflect tests with:
```python
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
```

- [ ] **Step 4: Run the suite — existing tests must still pass (no prod change yet)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: PASS. Note: the plan/reflect tests now pass `search_source` but production code still ignores it, so they pass exactly as before. This step proves the harness change is safe.

- [ ] **Step 5: Commit**

```bash
git add tests/test_nodes.py
git commit -m "test: record messages in _FakeModel; pass search_source in node tests"
```

---

## Task 2: `plan_node` source-specific prompts

**Files:**
- Modify: `deep_research/nodes.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_nodes.py`, add these two tests:
```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k plan_node -v`
Expected: FAIL — `test_plan_node_openalex_uses_keyword_prompt` fails because the current single prompt contains "子问题", not "关键词".

- [ ] **Step 3: Add `_PLAN_PROMPTS` and update `plan_node`**

In `deep_research/nodes.py`, add the prompt map near the top (after the imports) and update `plan_node`:
```python
_PLAN_PROMPTS = {
    "tavily": "你是研究规划助手。把用户的研究问题拆成 3-5 个互补、可独立检索的子问题。",
    "openalex": ("你是学术检索规划助手。把研究主题拆成 3-5 个精准的英文学术检索关键词组，"
                 "每组 2-5 个核心术语（领域术语 + 方法 + 场景），不要写成完整问句，"
                 "不要标点。例如：'OTFS anti-jamming deep reinforcement learning'。"),
}


def plan_node(state: ResearchState) -> dict:
    model = config.get_reasoning_model().with_structured_output(Plan)
    plan = model.invoke([
        SystemMessage(content=_PLAN_PROMPTS[state["search_source"]]),
        HumanMessage(content=state["topic"]),
    ])
    return {"sub_questions": plan.sub_questions, "iterations": 0}
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k plan_node -v`
Expected: PASS (3 plan tests: the original + 2 new).

- [ ] **Step 5: Commit**

```bash
git add deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: plan_node uses keyword prompt for openalex source"
```

---

## Task 3: `reflect_node` source-specific prompts

**Files:**
- Modify: `deep_research/nodes.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_nodes.py`, add these two tests:
```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k reflect_node -v`
Expected: FAIL — `test_reflect_node_openalex_uses_keyword_prompt` fails (current prompt has no "关键词").

- [ ] **Step 3: Add `_REFLECT_PROMPTS` and update `reflect_node`**

In `deep_research/nodes.py`, add the prompt map (near `_PLAN_PROMPTS`) and update `reflect_node`:
```python
_REFLECT_PROMPTS = {
    "tavily": "评估资料是否足够回答研究问题，并指出缺口。",
    "openalex": ("评估资料是否足够写一份学术综述，并指出缺口。若不足，"
                 "next_query 给出补充检索的精准英文学术关键词组（2-5 个术语，不要问句、不要标点）。"),
}


def reflect_node(state: ResearchState) -> dict:
    model = config.get_reasoning_model().with_structured_output(Reflection)
    digest = "\n".join(f"- {f['title']}: {f['content'][:200]}" for f in state["findings"])
    result = model.invoke([
        SystemMessage(content=_REFLECT_PROMPTS[state["search_source"]]),
        HumanMessage(content=f"研究问题：{state['topic']}\n\n已有资料：\n{digest}"),
    ])
    return {"reflection": "" if result.is_sufficient else result.next_query}
```

- [ ] **Step 4: Run to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k reflect_node -v`
Expected: PASS (4 reflect tests: 2 original + 2 new).

- [ ] **Step 5: Commit**

```bash
git add deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: reflect_node uses keyword prompt for openalex source"
```

---

## Task 4: `_dedupe_findings` + write_node dedup

**Files:**
- Modify: `deep_research/nodes.py`
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_nodes.py`, add these tests:
```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -k "dedupe or dedupes" -v`
Expected: FAIL — `nodes._dedupe_findings` does not exist; write_node currently emits `[2]` for the duplicate.

- [ ] **Step 3: Add `_dedupe_findings` and call it in `write_node`**

In `deep_research/nodes.py`, add the helper (just above `write_node`) and change write_node's first line:
```python
def _dedupe_findings(findings: list[dict]) -> list[dict]:
    """Dedupe by DOI (fall back to lowercased title); keep first occurrence, preserve order."""
    seen, out = set(), []
    for f in findings:
        key = f.get("doi") or (f.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(f)
    return out
```
And in `write_node`, change:
```python
    findings = state["findings"]
```
to:
```python
    findings = _dedupe_findings(state["findings"])
```
(Leave the rest of write_node unchanged — both `sources` and `_format_references(findings)` already use the local `findings` variable, so they automatically use the deduped list.)

- [ ] **Step 4: Run to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: PASS (all node tests, including the unchanged `test_write_node_appends_reference_list`).

- [ ] **Step 5: Commit**

```bash
git add deep_research/nodes.py tests/test_nodes.py
git commit -m "feat: dedupe findings by DOI/title before report generation"
```

---

## Task 5: Full suite, mocked E2E, real smoke re-run

**Files:** none (verification + the actual OTFS re-run the user requested)

- [ ] **Step 1: Run the full unit suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`
Expected: all tests pass, no network.

- [ ] **Step 2: Mocked end-to-end (no network) — confirm openalex keyword prompt + dedup wire together**

Run:
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
seq=[FM(structured=Plan(sub_questions=['OTFS anti-jamming'])),
     FM(structured=Reflection(is_sufficient=True,next_query='')),
     FM(content='# 综述\n[1] x')]
# fake search returns the SAME doi twice across the (single) query to test dedup
def fake_search(q, **k):
    return [{'query':q,'title':'P','url':'https://doi.org/10.1/x','content':'abs',
             'authors':['A'],'year':2021,'doi':'https://doi.org/10.1/x','venue':'IEEE','cited_by':10},
            {'query':q,'title':'P','url':'https://doi.org/10.1/x','content':'abs',
             'authors':['A'],'year':2021,'doi':'https://doi.org/10.1/x','venue':'IEEE','cited_by':10}]
with patch.object(config,'get_reasoning_model',lambda: seq.pop(0)), \
     patch.object(config,'get_summary_model',lambda: FM(content='摘要')), \
     patch.object(sp,'get_search_fn',lambda src: fake_search):
    from deep_research.graph import build_graph
    app=build_graph().compile()
    out=app.invoke({'topic':'T','max_iterations':1,'search_source':'openalex'})
    rpt=out['report']
    assert '## 参考文献' in rpt
    assert '[1]' in rpt and '[2]' not in rpt   # duplicate doi collapsed to one reference
    print('E2E OK; references deduped to single entry')
" 2>&1 | grep -vE "Warning|warn" | tail -5
```
Expected: prints `E2E OK; references deduped to single entry`.

- [ ] **Step 3: Real OTFS smoke re-run (requires network + model keys)**

Run (use a fresh thread-id to avoid checkpoint cache):
```bash
.venv/bin/python main.py "动态对抗环境下基于 OTFS 的智能抗干扰技术研究：(1) 基于深度强化学习的延迟-多普勒域资源跳跃；(2) 基于神经网络的 OTFS 接收机设计。重点关注低轨卫星(LEO)与无人机(UAV)之间的通信抗干扰" --source openalex --max-iters 5 --thread-id otfs-v3
```
Expected: node progress prints, report written. Then inspect the reference list:
```bash
F=$(ls -t reports/*.md | head -1)
grep -cE '^\[[0-9]+\]' "$F"          # count references
sed -n '/## 参考文献/,$p' "$F"        # show them
```
Acceptance: (a) no duplicate entries in the reference list; (b) at least one on-topic OTFS anti-jamming / DRL / neural-receiver primary paper appears (e.g. titles containing "OTFS" + "anti-jamming"/"jamming"/"receiver"/"detection"). If keys/network are unavailable, skip and note it.

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** A1 plan source prompt (T2) ✅; A2 reflect source prompt (T3) ✅; B dedup by DOI/title in write (T4) ✅; `_FakeModel` records messages + existing tests get `search_source` (T1, prevents KeyError) ✅; full suite + mocked E2E + real OTFS re-run acceptance (T5) ✅.
- **Type/name consistency:** `_PLAN_PROMPTS` / `_REFLECT_PROMPTS` keyed by the same `"tavily"`/`"openalex"` values that `--source` choices and the search dispatcher use. `_dedupe_findings(findings) -> list[dict]` called once in write_node; `Plan`/`Reflection` schemas unchanged; `state["search_source"]` read in plan_node and reflect_node (already present in ResearchState and graph inputs).
- **Ordering:** T1 must run first (test harness + existing-test fix) so T2/T3 prod changes don't KeyError on `search_source`. T1 makes no prod change, so its suite stays green.
- **No placeholders:** every step has complete code and exact commands with expected output.
