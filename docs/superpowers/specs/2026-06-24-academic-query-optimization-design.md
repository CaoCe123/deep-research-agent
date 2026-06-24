# 学术检索查询优化 + 去重 设计文档

- 日期：2026-06-24
- 状态：已确认（待实现）
- 关联：扩展 OpenAlex 学术检索能力（见 `2026-06-24-academic-search-openalex-design.md`）

## 1. 背景与目标

真实跑「OTFS 智能抗干扰」综述时发现两个质量问题：

1. **查询词不精准**：`plan_node` 为通用网页检索设计，生成的是长自然语言英文问句。OpenAlex 按相关性检索这种长问句时，召回的是泛主题综述（6G/ISAC），而非窄主题原始论文。诊断确认 OpenAlex **有**精准文献（"Spectrum Efficient Anti-Jamming for OTFS"、"Data-Driven Receiver for OTFS with Deep Learning" 等，"OTFS anti-jamming" 有 294 条），是查询策略把它们埋没了。
2. **跨轮重复**：多轮检索的结果直接累积，同一篇论文（甚至同一论文的 IEEE 版与 arXiv 版）多次出现在参考文献表。

**目标**：让 openalex 模式生成精准英文学术关键词组进行检索；报告生成前对文献去重。

**范围内：**
- `plan_node` 按 `search_source` 切换 prompt（openalex → 精准关键词组）。
- `reflect_node` 按 `search_source` 切换 prompt（openalex 的 `next_query` → 精准关键词组）。
- `write_node` 生成报告前按 DOI（无 DOI 用标题）去重 findings。

**范围外（YAGNI）：**
- 不改图结构、不改 `ResearchState`（`search_source` 已存在）。
- 不改检索源接口、不改 `make_finding`。
- 不在 search_node 做增量去重（统一在 write 前做）。
- 不引入新依赖。

## 2. 关键约束

`plan_node` 与 `reflect_node` 当前不读 `search_source`，但该字段已在 `ResearchState` 中（前一功能引入），且 `main.py` 已将其放入 graph inputs。因此两节点可直接 `state["search_source"]` 分支，无需改图或 state 结构。

## 3. 改动详情（全部在 `deep_research/nodes.py`）

### 3.1 plan_node 按源切 prompt

```python
_PLAN_PROMPTS = {
    "tavily": "你是研究规划助手。把用户的研究问题拆成 3-5 个互补、可独立检索的子问题。",
    "openalex": ("你是学术检索规划助手。把研究主题拆成 3-5 个精准的英文学术检索关键词组，"
                 "每组 2-5 个核心术语（领域术语 + 方法 + 场景），不要写成完整问句，"
                 "不要标点。例如：'OTFS anti-jamming deep reinforcement learning'。"),
}


def plan_node(state: ResearchState) -> dict:
    model = config.get_reasoning_model().with_structured_output(Plan)
    prompt = _PLAN_PROMPTS[state["search_source"]]
    plan = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["topic"]),
    ])
    return {"sub_questions": plan.sub_questions, "iterations": 0}
```
`Plan` schema 不变（`sub_questions: list[str]`）；openalex 模式下其语义从"子问题"变为"关键词组"。

### 3.2 reflect_node 按源切 prompt

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
`Reflection` schema 不变。

### 3.3 write 前去重

```python
def _dedupe_findings(findings: list[dict]) -> list[dict]:
    """按 DOI 去重（无 DOI 用小写标题）；保留首次出现，保序。"""
    seen, out = set(), []
    for f in findings:
        key = f.get("doi") or (f.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(f)
    return out
```
`write_node` 开头改为：
```python
def write_node(state: ResearchState) -> dict:
    findings = _dedupe_findings(state["findings"])
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
正文 `[n]` 与参考文献表均基于同一去重后列表，编号一致。`state["findings"]` 的累积（reducer）逻辑不变。

## 4. 错误处理

- `state["search_source"]` 不在 `_PLAN_PROMPTS`/`_REFLECT_PROMPTS` 时会 KeyError。该值来自 CLI `--source` 的 `choices=["tavily","openalex"]`，已被 argparse 限定为合法值，故不额外加防御（YAGNI）。
- `_dedupe_findings`：无 DOI 且无标题（空 key）的条目被跳过（不计入报告）——这类条目本就无引用价值。

## 5. 测试策略（TDD，全程 mock，无网络）

`tests/test_nodes.py` 调整：

- **`_FakeModel` 小改**：记录最后一次收到的 `messages`（新增 `self.last_messages`），以便断言 prompt 分支。
- **plan_node 按源切 prompt**（2 个新测试）：
  - openalex 源 → 捕获的 SystemMessage 含"关键词"。
  - tavily 源 → 含"子问题"。
- **reflect_node 按源切 prompt**（2 个新测试）：openalex 源含"关键词"；tavily 源含原文案。
- **现有 plan/reflect 测试**：补 `search_source` 字段（默认填 "tavily"），避免 KeyError。
- **`_dedupe_findings`**（3 个新测试）：
  - 同 DOI 去重为一条。
  - 无 DOI 时按标题（大小写不敏感）去重。
  - 不同 DOI 全保留且保序。
- **write_node 去重**（1 个新测试）：传入含重复 DOI 的 findings，断言报告参考文献表无重复（如 `report.count("[2]")` 行为符合预期 / 重复标题只出现一次）。

## 6. 验收标准

1. openalex 模式下 plan_node、reflect_node 使用关键词 prompt；tavily 模式 prompt 不变。
2. `_dedupe_findings` 正确按 DOI/标题去重并保序。
3. write_node 报告参考文献表无重复条目。
4. 全部单元测试通过，无网络依赖（现有 50 测试 + 新增不破坏旧测试）。
5. **真实冒烟验收**：重跑 OTFS 综述（`--source openalex --max-iters 5`），相比当前版本：(a) 参考文献无重复；(b) 命中此前缺失的 OTFS 抗干扰 / DRL 资源跳跃 / 神经网络 OTFS 接收机原始论文。
