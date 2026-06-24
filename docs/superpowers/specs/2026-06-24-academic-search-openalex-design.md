# 学术检索能力（OpenAlex）+ 文献综述 设计文档

- 日期：2026-06-24
- 状态：已确认（待实现）
- 关联：扩展现有的 Deep Research Agent（见 `2026-06-23-deep-research-agent-design.md`）

## 1. 目标与范围

为现有 Deep Research Agent 增加**专业学术论文检索能力**，使其能搜索到 IEEE（及其他出版商）论文的元数据与摘要，并产出**结构化文献综述 + 确定性生成的参考文献表**。

**核心诉求（用户确认）：**
- 能搜到 IEEE 论文（元数据 + 摘要即可，全文 PDF 在付费墙后，不在范围内）。
- 检索源可切换：跑的时候指定用 Tavily（网页）还是学术检索（OpenAlex）。
- 报告升级为结构化综述，末尾附参考文献表（含作者/年份/DOI/被引数）。

**范围内：**
- 新增 OpenAlex 学术检索源（无需 API key）。
- 检索源抽象：统一结果格式，CLI `--source` 切换。
- OpenAlex 倒排索引摘要还原。
- OpenAlex 结果按被引数降序（优先高被引论文）。
- `write_node` 产出综述结构 + 代码确定性生成的参考文献表。

**范围外（YAGNI）：**
- IEEE Xplore 全文 / 付费 API（用户确认元数据+摘要够用）。
- Semantic Scholar / Crossref（OpenAlex 已满足；实测 Semantic Scholar 无 key 限流 429）。
- 多源结果合并（本期是"二选一"切换，不是同时跑两个源再合并）。
- 复杂混合排序权重、年份过滤（本期只做被引降序）。

## 2. 学术检索源选型

实测三个免费 API（均索引 IEEE 论文）：

| API | IEEE 覆盖 | 摘要 | 被引数 | 免费稳定性 |
|---|---|---|---|---|
| **OpenAlex（选用）** | 9 万+ IEEE works | ✅ | ✅ | ✅ 无需 key、实测稳定 |
| Semantic Scholar | 有 | ✅ | ✅ | ❌ 无 key 时 HTTP 429 |
| Crossref | 43 万+ IEEE | ❌ 摘要常缺 | ✅ | ✅ |

**选 OpenAlex**：无需 API key、稳定、直接返回标题/作者/年份/DOI/被引数/摘要，支持 `cited_by_count:desc` 排序。

OpenAlex 端点（无需 key）：
```
GET https://api.openalex.org/works?search=<query>&sort=cited_by_count:desc&per-page=<n>
```
建议带上 `mailto` 参数进入 "polite pool"（更稳定的速率）：`&mailto=deep-research-agent@example.com`。

## 3. 架构与文件布局

新增 1 个目录（4 文件），改动 3 个文件，`tools.py`/`graph.py`/`config.py` 行为不变。

```
deep_research/
├── tools.py          # 不动：slugify（tavily_search 迁出，见下）
├── search/           # 新增：检索源
│   ├── __init__.py   # 导出 get_search_fn(source) 调度器
│   ├── base.py       # 统一结果结构 make_finding() + SOURCES 注册表
│   ├── tavily.py     # 从 tools.tavily_search 迁移而来，输出统一格式
│   └── openalex.py   # OpenAlex 检索 + 倒排索引摘要还原
├── state.py          # 改：findings 增加学术字段；新增 search_source
├── nodes.py          # 改：search_node 按 search_source 调度；write_node 生成参考文献表
└── graph.py          # 不动
main.py               # 改：新增 --source 参数 + 校验
```

> 迁移说明：`tools.tavily_search` 移到 `search/tavily.py` 并改造为统一格式。为避免破坏 Task 3 的现有测试 `tests/test_tools.py`，在 `tools.py` 保留一个**向后兼容的薄包装**（`tavily_search = search.tavily.search` 的旧式签名），或同步更新该测试。实现时采用：更新 `tests/test_tools.py` 让 slugify 测试保留、tavily 测试迁移到 `tests/test_search_tavily.py`。

## 4. 统一检索结果格式（base.py）

所有检索源返回同样结构的 `dict`，下游节点无差别处理：

```python
def make_finding(query, title, url, content,
                 authors=None, year=None, doi=None, venue=None, cited_by=None) -> dict:
    return {
        "query": query,
        "title": title,
        "url": url,
        "content": content,        # Tavily=网页正文；OpenAlex=摘要
        "authors": authors or [],  # 学术源专属，Tavily 留空
        "year": year,
        "doi": doi,
        "venue": venue,
        "cited_by": cited_by,
    }
```

调度器：

```python
# search/__init__.py
SOURCES = {"tavily": ..., "openalex": ...}  # name -> search 函数

def get_search_fn(source: str):
    if source not in SOURCES:
        raise ValueError(f"未知检索源: {source}；可选: {', '.join(SOURCES)}")
    return SOURCES[source]
```

每个源的函数签名统一：`search(query: str, max_results: int = 3) -> list[dict]`。

## 5. OpenAlex 检索（openalex.py）

```python
import urllib.parse, urllib.request, json
from .base import make_finding

OPENALEX_URL = "https://api.openalex.org/works"

def _restore_abstract(inverted_index: dict | None) -> str:
    """OpenAlex 返回倒排索引形式的摘要，还原成正文。"""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)

def search(query: str, max_results: int = 3) -> list[dict]:
    params = {
        "search": query,
        "sort": "cited_by_count:desc",   # 优先高被引
        "per-page": max_results,
        "mailto": "deep-research-agent@example.com",
    }
    url = f"{OPENALEX_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — 单次失败不中断整轮
        print(f"[warn] OpenAlex 检索失败 query={query!r}: {e}")
        return []
    findings = []
    for w in data.get("results", []):
        authors = [a["author"]["display_name"]
                   for a in w.get("authorships", []) if a.get("author")]
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        findings.append(make_finding(
            query=query,
            title=w.get("title") or "",
            url=w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url") or "",
            content=_restore_abstract(w.get("abstract_inverted_index")),
            authors=authors,
            year=w.get("publication_year"),
            doi=w.get("doi"),
            venue=venue,
            cited_by=w.get("cited_by_count"),
        ))
    return findings
```

> 用标准库 `urllib`，不引入新依赖。IEEE 论文会自然出现在结果里（IEEE 是 OpenAlex 索引的出版商之一）；用户若要严格限定 IEEE，可在 query 中包含来源线索，但本期不做硬过滤（避免过度设计，且会牺牲召回）。

## 6. Tavily 检索迁移（tavily.py）

把现有 `tools.tavily_search` 迁移到 `search/tavily.py`，改为输出统一格式（学术字段留空）：

```python
from tavily import TavilyClient
from .base import make_finding

def search(query: str, max_results: int = 3) -> list[dict]:
    try:
        client = TavilyClient()
        resp = client.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 检索失败 query={query!r}: {e}")
        return []
    return [make_finding(query=query, title=r.get("title", ""),
                         url=r.get("url", ""), content=r.get("content", ""))
            for r in resp.get("results", [])]
```

`tools.py` 仅保留 `slugify`。`tests/test_tools.py` 中的 tavily 测试迁移到 `tests/test_search_tavily.py`，slugify 测试保留在原处。

## 7. 状态变更（state.py）

`ResearchState` 增加一个字段：

```python
class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]  # dict 现含学术字段（见 base.make_finding）
    reflection: str
    iterations: int
    max_iterations: int
    search_source: str        # 新增："tavily" | "openalex"
    report: str
```

`findings` 的元素结构由 `make_finding` 统一定义，无需改 reducer。

## 8. 节点变更（nodes.py）

### search_node
```python
from .search import get_search_fn

def search_node(state):
    search_fn = get_search_fn(state["search_source"])
    queries = state["sub_questions"] if state["iterations"] == 0 else [state["reflection"]]
    new_findings = []
    for q in queries:
        for hit in search_fn(q):
            hit = dict(hit)
            hit["content"] = _summarize(hit["content"]) if hit["content"] else ""
            new_findings.append(hit)
    return {"findings": new_findings, "iterations": state["iterations"] + 1}
```
（`plan_node` / `reflect_node` 不变。）

### write_node — 综述 + 确定性参考文献表
```python
def _format_references(findings: list[dict]) -> str:
    lines = ["## 参考文献"]
    for i, f in enumerate(findings, 1):
        if f.get("doi") or f.get("authors"):   # 学术源
            authors = ", ".join(f["authors"][:3]) + (" 等" if len(f["authors"]) > 3 else "")
            cite = f"（被引 {f['cited_by']} 次）" if f.get("cited_by") is not None else ""
            parts = [p for p in [authors, f["title"],
                                 f.get("venue"), str(f.get("year") or "")] if p]
            ref = ". ".join(parts)
            if f.get("doi"):
                ref += f". DOI:{f['doi']}"
            lines.append(f"[{i}] {ref} {cite}".rstrip())
        else:                                    # Tavily 源降级
            lines.append(f"[{i}] {f['title']} — {f['url']}")
    return "\n".join(lines)

def write_node(state):
    findings = state["findings"]
    sources = "\n".join(
        f"[{i+1}] {f['title']}（{f.get('venue') or 'web'}, {f.get('year') or 'n.d.'}）\n{f['content'][:500]}"
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
    body = resp.content
    references = _format_references(findings)
    return {"report": f"{body}\n\n{references}"}
```

**关键：参考文献表由代码确定性拼接**（从 findings 的学术字段），不依赖 LLM，杜绝引用幻觉。正文 `[n]` 编号与 findings 顺序、参考文献表编号一致。

## 9. CLI 变更（main.py）

新增 `--source`：
```python
parser.add_argument("--source", default="tavily", choices=["tavily", "openalex"],
                    help="检索源：tavily（网页）| openalex（学术论文）")
```
- `argparse` 的 `choices` 已能拦截未知值（自动报错退出，非 0）。
- 传入 `inputs` 时带上 `"search_source": args.source`。

## 10. 错误处理

- OpenAlex / Tavily 请求失败 → 返回 `[]`，单次失败不中断整轮（与现有一致）。
- 倒排索引摘要为空/异常 → content 降级为空串，`search_node` 跳过摘要。
- `--source` 未知值 → argparse 自动报错退出（非 0）。
- 参考文献缺字段（无作者/无被引）→ 降级格式，不崩。

## 11. 测试策略（延续 TDD，全程 mock 网络）

**单元测试：**
- `tests/test_search_base.py`：`make_finding` 字段完整性；`get_search_fn` 未知源抛 `ValueError`。
- `tests/test_search_openalex.py`：mock `urllib.request.urlopen`，验证①倒排索引摘要还原顺序正确 ②字段映射（authors/venue/doi/cited_by）正确 ③异常返回 `[]` ④请求 URL 含 `sort=cited_by_count:desc`。
- `tests/test_search_tavily.py`：mock TavilyClient，验证统一格式输出 + 异常返回 `[]`（迁移自原 test_tools.py）。
- `tests/test_tools.py`：仅保留 slugify 测试。
- `tests/test_nodes.py`：search_node 按 `search_source` 调度到正确的源（mock get_search_fn）；write_node 的参考文献表为代码生成（学术源含 DOI/被引；Tavily 源降级格式）。
- `tests/test_cli.py`：`--source` 默认 tavily、可设 openalex、未知值退出非 0。

**真实冒烟跑（需网络，无需额外 key）：**
```bash
.venv/bin/python main.py "deep learning for wireless communication" --source openalex --max-iters 1
```
确认：端到端跑通；报告含「参考文献」段；参考文献含 DOI/被引数；结果里出现 IEEE 来源论文。

## 12. 验收标准

1. `--source openalex` 端到端跑通，生成含参考文献表的综述报告，能检索到 IEEE 论文。
2. `--source tavily`（默认）行为与现状一致，无回归。
3. OpenAlex 倒排索引摘要正确还原为可读正文。
4. OpenAlex 结果按被引数降序。
5. 参考文献表由代码确定性生成（学术源含作者/年份/DOI/被引；Tavily 源降级为标题+URL），正文 `[n]` 与之对应。
6. `--source` 传未知值时报错退出（非 0）。
7. 全部单元测试通过（含迁移后的测试），无网络依赖。
