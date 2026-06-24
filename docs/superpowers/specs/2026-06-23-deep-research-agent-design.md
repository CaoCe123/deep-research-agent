# Deep Research Agent 设计文档

- 日期：2026-06-23
- 状态：已确认（待实现）
- 形态：可复用的本地命令行工具

## 1. 目标与范围

用 LangGraph 从零搭建一个 Deep Research Agent，以「可复用的本地工具」形态交付：
传入一个研究问题，Agent 自动完成 `规划 → 检索 → 反思 →（条件循环）→ 写作`，
最终产出一份带编号引用的 Markdown 报告文件。

**范围内：**
- 四节点研究闭环（plan / search / reflect / write）+ 反思驱动的检索循环 + 轮数熔断
- Tavily 网络检索
- 混合模型（推理用 Opus 4.8，检索摘要用 Haiku）
- SQLite checkpointer 落盘（可断点续跑）
- 薄 CLI 入口，流式打印进度，报告写入 `reports/`
- 可选 LangSmith 追踪（环境变量开启，零侵入）
- 控制流与数据契约的单元测试 + 一次真实冒烟跑

**范围外（YAGNI）：**
- 人在回路 / 审批（确认为全自动）
- Web 服务 / API / 并发部署
- 本地文档 RAG（仅网络检索）
- 多用户、鉴权、前端界面

## 2. 架构与文件布局

```
text1/
├── deep_research/
│   ├── __init__.py
│   ├── config.py      # 模型配置：Opus 4.8(规划/反思/写作) + Haiku(检索摘要)
│   ├── state.py       # ResearchState (TypedDict) + reducers
│   ├── tools.py       # Tavily 检索封装 + 错误处理
│   ├── nodes.py       # plan / search / reflect / write 四个节点
│   └── graph.py       # 组图 + 条件边 + SQLite checkpointer + compile
├── main.py            # CLI 入口
├── requirements.txt
├── reports/           # 产出的 Markdown 报告（运行时生成）
└── research.sqlite    # checkpointer 落盘（运行时生成）
```

**选型理由：** 扁平模块 + 薄 CLI，模块少、边界清晰，契合「可复用本地工具」的规模；
比完整分层包（`src/deep_research/nodes/...`）更符合 YAGNI，后续要拆分也容易。

**数据流：**
```
CLI → graph.stream/invoke → plan → search →（reflect 条件路由）→ search↺ 或 write → 写 reports/<slug>.md
```

## 3. 状态设计（state.py）

`ResearchState(TypedDict)`：

| 字段 | 类型 | reducer | 说明 |
|---|---|---|---|
| `topic` | `str` | 覆盖 | 原始研究问题 |
| `sub_questions` | `list[str]` | 覆盖 | plan 拆出的 3-5 个子问题 |
| `findings` | `Annotated[list[dict], operator.add]` | 追加 | 每轮检索结果累积，每条含 `query / title / url / content` |
| `reflection` | `str` | 覆盖 | 反思给出的下一步查询；空串表示信息已足够 |
| `iterations` | `int` | 覆盖 | 已检索轮数 |
| `max_iterations` | `int` | 覆盖 | 轮数熔断上限（CLI 传入，默认 3） |
| `report` | `str` | 覆盖 | 最终 Markdown 报告 |

`findings` 用 `operator.add` 作为 reducer，实现多轮检索结果的累积（追加而非覆盖）。

## 4. 节点设计（nodes.py）

每个节点只做一件事，输入 state、输出对 state 的更新（dict）。

1. **plan_node** — `claude-opus-4-8` + `with_structured_output(Plan)`。
   把 `topic` 拆成 3-5 个互补子问题。返回 `{sub_questions, iterations: 0}`。
   - `Plan(BaseModel)`: `sub_questions: list[str]`

2. **search_node** — Tavily 检索 + `claude-haiku-4-5-20251001` 摘要。
   - 首轮（`iterations == 0`）检索 `sub_questions`；后续轮检索 `reflection` 给出的查询。
   - 对每条来源用 Haiku 压成要点，降低后续 token 消耗。
   - 返回 `{findings: [...新结果...], iterations: iterations + 1}`（经 reducer 追加）。

3. **reflect_node** — `claude-opus-4-8` + `with_structured_output(Reflection)`。
   - 基于当前 `findings` 摘要，判断信息是否足以写报告。
   - `Reflection(BaseModel)`: `is_sufficient: bool`, `next_query: str`
   - 返回 `{reflection: next_query if not is_sufficient else ""}`。

4. **write_node** — `claude-opus-4-8`。
   - 综合所有 `findings`，产出结构化、带编号引用 `[1][2]...` 的 Markdown 报告。
   - 返回 `{report: ...}`。

## 5. 图与控制流（graph.py）

```
START → plan → search → reflect →（条件边 route_after_reflect）
                                   ├── "search" → search   （回环：信息不足且未到上限）
                                   └── "write"  → write → END
```

**条件边 `route_after_reflect(state) -> str`：**
- `state["reflection"] == ""` → `"write"`（反思认为信息足够）
- `state["iterations"] >= state["max_iterations"]` → `"write"`（轮数熔断）
- 否则 → `"search"`（继续检索，形成循环）

**编译：** 使用 `langgraph-checkpoint-sqlite` 的 `SqliteSaver`，落盘到 `research.sqlite`，
配合 `thread_id` 支持断点续跑与历史查看。

## 6. 模型配置（config.py）

集中定义，温度 0：
- `REASONING_MODEL = "claude-opus-4-8"` — 规划 / 反思 / 写作
- `SUMMARY_MODEL = "claude-haiku-4-5-20251001"` — 检索结果摘要

提供工厂函数返回配置好的 `ChatAnthropic` 实例，便于测试时替换。

## 7. CLI 行为（main.py）

```
python main.py "研究问题" [--max-iters 3] [--out reports/] [--thread-id auto]
```
- 用 `graph.stream(..., stream_mode="updates")` 逐节点打印进度（如 `== plan 完成 ==`）。
- 结束后将 `report` 写入 `reports/<slug>.md`；`slug` 由 topic 生成（保留中英文与数字，其余替换为 `-`，截断长度）。
- `--thread-id` 默认按 topic 生成稳定 id，便于同题续跑；显式传入可覆盖。
- 退出码：成功 0；缺少必需 API key 或最终无产出时非 0。

## 8. 依赖与配置

`requirements.txt`：
```
langgraph
langchain
langchain-anthropic
langgraph-checkpoint-sqlite
tavily-python
```

API key 全部从环境变量读取，不写入代码：
- `ANTHROPIC_API_KEY`（必需）
- `TAVILY_API_KEY`（必需）
- `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`（可选，开启追踪）

## 9. 错误处理

- **Tavily 调用**：包 try/except，单条查询失败只记录并跳过，不中断整轮检索。
- **启动检查**：缺 `ANTHROPIC_API_KEY` 或 `TAVILY_API_KEY` 时，启动即友好报错并以非 0 退出。
- **结构化输出**：失败由 LangChain 自带重试兜底。
- **空报告**：若 `findings` 为空或写作产出为空，提示"资料不足"并非 0 退出。
- **LangSmith**：未配置时不报错，仅无 trace（零侵入）。

## 10. 测试策略

LLM 输出不确定，因此单元测试聚焦**确定性的控制流与数据契约**，LLM 与 Tavily 一律 mock。

**单元测试（pytest，不烧钱）：**
- `route_after_reflect` 三个分支：信息足够 → write；到达上限 → write；否则 → search。
- `findings` 的 `operator.add` 累积行为（多次更新后是追加而非覆盖）。
- slug 生成（特殊字符处理、长度截断）与 CLI 参数解析。
- 节点对 state 的读写契约：mock 掉 LLM / Tavily，验证各节点返回的 dict 字段正确。

**真实冒烟跑（需要 key，手动执行一次）：**
- 跑一个小问题、`--max-iters 1`，确认端到端通畅并生成报告文件。
- 质量与推理过程通过 LangSmith trace 人工核验。

## 11. 验收标准

1. `python main.py "..." --max-iters 1` 端到端跑通，在 `reports/` 生成带编号引用的 Markdown 报告。
2. 反思循环可触发第二轮检索；达到 `max_iterations` 时正确熔断进入写作。
3. 单元测试全部通过（路由三分支、reducer 累积、slug、参数解析、节点契约）。
4. 缺少 API key 时给出清晰报错并非 0 退出。
5. 未配置 LangSmith 时仍能正常运行。
