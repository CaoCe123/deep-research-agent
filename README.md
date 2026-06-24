# Deep Research Agent

用 **LangGraph** 从零搭建的深度研究（Deep Research）Agent，并附带一份
**LangChain / LangGraph / LangSmith** 的详细中文教学文档。

传入一个研究问题，Agent 自动完成 `规划 → 检索 → 反思 →（条件循环）→ 写作`，
最终产出一份带编号引用的 Markdown 报告。

## 内容

- 📘 [LangChain / LangGraph / LangSmith 教学文档](./LangChain-LangGraph-LangSmith-教学文档.md)
  —— 三件套定位、Deep Research 实战、与 OpenClaw / Claude Code 的对比。
- 📐 [Deep Research Agent 设计文档](./docs/superpowers/specs/2026-06-23-deep-research-agent-design.md)
  —— 架构、状态、节点、控制流、测试策略与验收标准。

## 设计概览

```
START → plan → search → reflect →（条件路由）
                                   ├── 信息不足且未到上限 → search（循环）
                                   └── 足够 / 达到熔断上限 → write → END
```

- **检索**：Tavily（含单条失败跳过的容错）
- **模型**：规划/反思/写作用 `claude-opus-4-8`，检索摘要用 `claude-haiku-4-5-20251001`
- **持久化**：SQLite checkpointer 落盘，支持断点续跑
- **观测**：可选 LangSmith 追踪（环境变量开启，零侵入）

## 用法

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# API key —— 写到环境变量 ANTHROPIC_API_KEY（agibot 的 bearer token 也填这里）
export ANTHROPIC_API_KEY="<你的-agibot-bearer-token>"
export TAVILY_API_KEY="tvly-..."
# 可选：开启 LangSmith 追踪
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="deep-research-agent"

.venv/bin/python main.py "你的研究问题" --max-iters 3
# 报告写入 reports/<slug>.md
```

### Provider 配置（endpoint）

模型的 endpoint 等信息在仓库根目录的 [`providers.json`](./providers.json) 中配置，
当前指向 agibot 的 `https://lingzhi.agibot.com`（Anthropic 协议、bearer 鉴权）。
`config.py` 会读取其中的 `baseURL` 并传给 `ChatAnthropic`。

> **API key 写在哪里？** 不写进 `providers.json`，而是放到环境变量 **`ANTHROPIC_API_KEY`**
> （上面的 `export` 那一行）。agibot 的 bearer token 直接填到这个变量即可——
> `ChatAnthropic` 会以 bearer 方式带上它访问 `baseURL`。

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `topic`（位置参数） | — | 研究问题 |
| `--max-iters` | `3` | 检索轮数上限（熔断） |
| `--out` | `reports` | 报告输出目录 |
| `--thread-id` | 按 topic 生成 | checkpointer 线程 id，同 id 可断点续跑 |
| `--sqlite` | `research.sqlite` | checkpointer 落盘路径 |

## 开发

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -v
```

## 许可证

MIT
