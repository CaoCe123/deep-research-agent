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

# 复制模板并填入真实 key（.env 已被 .gitignore 忽略，不会提交）
cp .env.example .env
# 然后编辑 .env，填入 ANTHROPIC_API_KEY / TAVILY_API_KEY

.venv/bin/python main.py "你的研究问题" --max-iters 3
# 报告写入 reports/<slug>.md
```

程序启动时会自动加载项目根目录的 `.env`（通过 python-dotenv）。
也可以不用 `.env`，直接 `export` 这些环境变量——两种方式等价：

```bash
export ANTHROPIC_API_KEY="<你的-agibot-bearer-token>"
export TAVILY_API_KEY="tvly-..."
```

`.env` 中可配置的变量见 [`.env.example`](./.env.example)：`ANTHROPIC_API_KEY`、
`TAVILY_API_KEY`，以及可选的 `LANGSMITH_*` 追踪开关。

### Provider 配置（endpoint）

模型的 endpoint 等信息在仓库根目录的 [`providers.json`](./providers.json) 中配置，
当前指向 agibot 的 `https://lingzhi.agibot.com`（Anthropic 协议、bearer 鉴权）。
`config.py` 会读取其中的 `baseURL` 并传给 `ChatAnthropic`。

> **API key 写在哪里？** 不写进 `providers.json`，而是放到 `.env` 里的 **`ANTHROPIC_API_KEY`**
> （或同名环境变量）。agibot 的 bearer token 直接填到这个变量即可——
> `ChatAnthropic` 会以 bearer 方式带上它访问 `baseURL`。

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `topic`（位置参数） | — | 研究问题 |
| `--max-iters` | `3` | 检索轮数上限（熔断） |
| `--out` | `reports` | 报告输出目录 |
| `--thread-id` | 按 topic 生成 | checkpointer 线程 id，同 id 可断点续跑 |
| `--sqlite` | `research.sqlite` | checkpointer 落盘路径 |
| `--source` | `tavily` | 检索源：`tavily`（网页）/ `openalex`（学术论文，可搜 IEEE 等） |

### 学术检索（OpenAlex）

`--source openalex` 切换到学术论文检索，基于 [OpenAlex](https://openalex.org)（无需 API key），
可检索到 IEEE 等出版商论文的标题/作者/年份/DOI/被引数/摘要，并按被引数降序优先纳入高被引论文。
此时报告升级为结构化文献综述，末尾附**由系统确定性生成的参考文献表**（不依赖模型，杜绝引用幻觉）。

```bash
.venv/bin/python main.py "deep learning for wireless communication" --source openalex --max-iters 3
```

> 注：免费方案只能获取论文的元数据与摘要；IEEE 等的全文 PDF 在付费墙后，不在覆盖范围内。

## 开发

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -v
```

## 许可证

MIT
