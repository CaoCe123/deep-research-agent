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

## 计划中的用法

> 实现尚在进行中（当前仓库已包含教学文档与设计文档）。

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
export TAVILY_API_KEY="tvly-..."
# 可选：开启 LangSmith 追踪
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."

python main.py "你的研究问题" --max-iters 3
```

## 许可证

MIT
