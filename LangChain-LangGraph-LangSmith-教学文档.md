# LangChain / LangGraph / LangSmith 教学文档

> 以「构建一个 Deep Research（深度研究）Agent」为主线，详细讲解三件套的定位、用法与协作方式，
> 并对比「用框架写 Agent」与「用 OpenClaw / Claude Code 这类现成 Agent CLI 做 research」的本质区别。

---

## 目录

1. [三件套总览：它们各自解决什么问题](#1-三件套总览)
2. [核心概念逐个拆解](#2-核心概念逐个拆解)
   - 2.1 LangChain
   - 2.2 LangGraph
   - 2.3 LangSmith
3. [什么是 Deep Research Agent](#3-什么是-deep-research-agent)
4. [环境准备](#4-环境准备)
5. [实战：用 LangGraph 从零搭一个 Deep Research Agent](#5-实战用-langgraph-从零搭一个-deep-research-agent)
   - 5.1 状态设计
   - 5.2 节点：规划 / 检索 / 反思 / 写作
   - 5.3 用图把节点连起来（含循环）
   - 5.4 运行与流式输出
6. [用 LangSmith 做可观测、评测与调试](#6-用-langsmith-做可观测评测与调试)
7. [对比：框架自研 Agent vs OpenClaw / Claude Code](#7-对比框架自研-agent-vs-openclaw--claude-code)
8. [选型建议与常见误区](#8-选型建议与常见误区)
9. [附录：术语表与参考链接](#9-附录)

---

## 1. 三件套总览

LangChain 生态在 2024–2026 年经历了一次明显的「分层」，现在最清晰的理解方式是：**三个库分别负责"组件、编排、观测"三件事**。

| 库 | 一句话定位 | 解决的核心问题 | 类比 |
|---|---|---|---|
| **LangChain** | LLM 应用的「标准组件库」 | 统一各家模型 / 工具 / 检索器的接口，提供可拼装的积木 | 标准库 / SDK |
| **LangGraph** | 有状态、可循环的「Agent 编排引擎」 | 用图（状态机）描述复杂、带循环和分支的 Agent 控制流，支持持久化、人在回路、中断恢复 | 工作流引擎 / 状态机框架 |
| **LangSmith** | 「可观测 + 评测」平台 | 把每一次 LLM 调用、每一步推理都记录下来，可追踪、可评测、可回放 | APM / 监控 + 测试平台 |

关键认知：

- **三者不是层层依赖的关系，而是「可单独使用、组合更香」**。
  - 你可以只用 LangChain 的模型封装，不用图。
  - 你可以用 LangGraph 编排，但模型调用直接用厂商 SDK。
  - 你可以给任何代码（哪怕完全不用 LangChain）加上 LangSmith 追踪——它只是一个装饰器 / 回调。
- **2026 年的主流建议：Agent 类应用用 LangGraph 编排，LangChain 提供组件，LangSmith 兜底观测。**
  早期 LangChain 里的 `AgentExecutor`、`initialize_agent` 等"黑盒 Agent"已被官方标记为遗留方案，复杂控制流一律推荐 LangGraph。

```
┌─────────────────────────────────────────────┐
│                你的 Agent 应用                 │
│                                               │
│   控制流 / 状态 / 循环  ──►  LangGraph         │
│   模型 / 工具 / 检索器  ──►  LangChain 组件     │
│   追踪 / 评测 / 调试    ──►  LangSmith（旁路）  │
└─────────────────────────────────────────────┘
```

---

## 2. 核心概念逐个拆解

### 2.1 LangChain

LangChain 的本质是**抽象 + 组件**。它把"调用一个 LLM 应用"里反复出现的东西标准化：

| 组件 | 作用 | 典型类 |
|---|---|---|
| **Chat Model** | 统一不同厂商的对话模型接口 | `ChatAnthropic`、`ChatOpenAI`、`init_chat_model(...)` |
| **Messages** | 标准化对话消息 | `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` |
| **Prompt Template** | 参数化提示词 | `ChatPromptTemplate` |
| **Tools** | 让模型可以"调用函数" | `@tool` 装饰器 |
| **Output Parser** | 把模型输出解析成结构化数据 | `with_structured_output(...)`（推荐，基于 function calling） |
| **Retriever / VectorStore** | RAG 检索 | `Chroma`、`FAISS`、`.as_retriever()` |
| **LCEL** | 用 `|` 把组件串成链 | `prompt | model | parser` |

#### LCEL（LangChain Expression Language）

LCEL 用管道符把组件组合成"链"，所有链统一支持 `invoke / stream / batch / ainvoke` 等接口：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatAnthropic(model="claude-opus-4-8")  # 使用最新最强的 Claude 模型
prompt = ChatPromptTemplate.from_template("用一句话解释这个概念：{topic}")

chain = prompt | model | StrOutputParser()   # 这就是一条 LCEL 链
print(chain.invoke({"topic": "向量数据库"}))
```

**LCEL 适合"线性、无状态、走一遍就结束"的流程**（比如 RAG 问答、文本改写、分类）。
一旦需要循环、分支、记忆、人工介入，就该上 LangGraph。

#### 结构化输出（Deep Research 里非常关键）

```python
from pydantic import BaseModel, Field

class SearchPlan(BaseModel):
    sub_questions: list[str] = Field(description="把研究问题拆分成 3-5 个子问题")
    rationale: str = Field(description="拆分理由")

planner = model.with_structured_output(SearchPlan)
plan = planner.invoke("研究：2026 年小型语言模型在边缘设备上的部署现状")
print(plan.sub_questions)
```

---

### 2.2 LangGraph

LangGraph 把 Agent 建模成一张**有向图（图 = 状态机）**：

- **State（状态）**：一个贯穿全图的共享数据结构（通常是 `TypedDict`）。每个节点读它、改它。
- **Node（节点）**：一个普通函数 / 可调用对象，输入是 state，输出是"对 state 的更新"。
- **Edge（边）**：节点之间的连接。
  - 普通边：A 做完一定走 B。
  - **条件边（conditional edge）**：根据 state 的内容决定下一步去哪 —— 这是实现"循环 / 反思 / 重试"的关键。
- **Reducer（归约器）**：定义某个 state 字段如何被多次更新合并（例如消息列表用 `add_messages` 追加而不是覆盖）。
- **Checkpointer（检查点）**：把每一步 state 持久化，支持「中断后恢复」「时间旅行」「人在回路」。

为什么 Agent 要用图而不是简单 while 循环？因为真实 Agent 需要：

1. **循环**：检索 → 反思「够不够」→ 不够再检索（次数还得有上限）。
2. **分支**：信息足够就去写作，不足就继续搜。
3. **持久化与恢复**：长任务跑一半断了能续上；人工审批后再继续。
4. **可观测**：每个节点边界天然就是一个可追踪、可调试的步骤。

把这些用裸 `while/if` 写也能跑，但很快会变成一团乱麻。LangGraph 把"控制流"显式化、可视化、可持久化。

> **快捷方式**：如果只是想要一个"会用工具的标准 ReAct Agent"，LangGraph 预置了
> `from langgraph.prebuilt import create_react_agent`，一行就能拿到一个带工具循环的 Agent。
> 但 Deep Research 通常需要自定义控制流（规划→检索→反思→写作），所以本文走"手搓图"的路线，便于理解原理。

---

### 2.3 LangSmith

LangSmith 是**旁路**的（不侵入你的业务逻辑），主要做三件事：

1. **Tracing（追踪）**：自动记录每一次 LLM 调用的输入/输出/token/耗时/工具调用，形成一棵可展开的调用树。对调试 Agent"为什么走错路"极其有用。
2. **Evaluation（评测）**：用数据集 + 评测器（可以是规则、也可以是"LLM as judge"）批量打分，量化"改了 prompt 之后到底变好还是变差"。
3. **Monitoring（线上监控）**：上线后看延迟、成本、错误率、用户反馈。

最妙的是：**开启追踪几乎零成本**，只要设几个环境变量：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="deep-research-agent"
```

设好之后，所有 LangChain / LangGraph 的执行会自动上报。**即使代码里完全不 import langsmith，也照样有 trace**。对非 LangChain 代码，则用 `@traceable` 装饰器手动接入。

---

## 3. 什么是 Deep Research Agent

"Deep Research"（深度研究）指的不是一次性问答，而是一个**多步骤、多来源、带自我反思**的研究流程，典型形态：

```
研究问题
   │
   ▼
①规划 Plan ── 把大问题拆成若干子问题 / 检索策略
   │
   ▼
②检索 Search ── 对每个子问题做网络/文档检索，抓取并阅读来源
   │
   ▼
③反思 Reflect ── 评估「已有信息是否足够回答？还有什么缺口？」
   │
   ├── 信息不足 ──► 生成新的检索词，回到 ②（循环，有上限）
   │
   └── 信息充足 ──►
   ▼
④写作 Write ── 综合所有来源，产出带引用的结构化报告
```

它的难点正好对应 LangGraph 的强项：**循环 + 条件分支 + 状态累积 + 可观测**。

---

## 4. 环境准备

```bash
pip install -U langgraph langchain langchain-anthropic langsmith
pip install -U tavily-python    # 一个常用的检索 API（也可换成别的）

export ANTHROPIC_API_KEY="sk-ant-..."
export TAVILY_API_KEY="tvly-..."

# 开启 LangSmith 追踪（强烈建议）
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls-..."
export LANGSMITH_PROJECT="deep-research-agent"
```

> 模型方面，本文统一用 Anthropic 最新最强的 **Claude Opus 4.8**（model id：`claude-opus-4-8`）。
> 需要更省钱/更快时可换 `claude-sonnet-4-6` 或 `claude-haiku-4-5-20251001`。

---

## 5. 实战：用 LangGraph 从零搭一个 Deep Research Agent

下面是一份**可读、可逐段理解**的完整实现。为了聚焦原理，做了适度简化（真实生产还要加错误处理、超时、去重、来源缓存等）。

### 5.1 状态设计

State 是整个图的"共享内存"。我们让它累积子问题、检索结果和循环计数。

```python
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    topic: str                                   # 原始研究问题
    sub_questions: list[str]                     # 规划出的子问题
    # operator.add 作为 reducer：每轮检索的结果"追加"而非"覆盖"
    findings: Annotated[list[dict], operator.add]
    reflection: str                              # 最近一次反思结论
    iterations: int                              # 已经检索的轮数
    max_iterations: int                          # 轮数上限，防止无限循环
    report: str                                  # 最终报告
```

> `Annotated[list[dict], operator.add]` 是 LangGraph 的精髓之一：
> 它告诉图——当多个节点（或多轮）都往 `findings` 写时，用「相加（拼接）」来合并，
> 而不是后者覆盖前者。这就是"状态累积"的实现方式。

### 5.2 节点：规划 / 检索 / 反思 / 写作

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tavily import TavilyClient

llm = ChatAnthropic(model="claude-opus-4-8", temperature=0)
search_client = TavilyClient()

# ---------- ① 规划节点 ----------
class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3-5 个可独立检索的子问题")

def plan_node(state: ResearchState) -> dict:
    planner = llm.with_structured_output(Plan)
    result = planner.invoke([
        SystemMessage(content="你是研究规划助手。把用户的研究问题拆成 3-5 个互补的子问题。"),
        HumanMessage(content=state["topic"]),
    ])
    return {"sub_questions": result.sub_questions, "iterations": 0}

# ---------- ② 检索节点 ----------
def search_node(state: ResearchState) -> dict:
    # 第一轮检索 sub_questions；后续轮次检索"反思"给出的新查询
    queries = state["sub_questions"] if state["iterations"] == 0 \
              else [state["reflection"]]
    new_findings = []
    for q in queries:
        resp = search_client.search(q, max_results=3)
        for r in resp["results"]:
            new_findings.append({
                "query": q,
                "title": r["title"],
                "url": r["url"],
                "content": r["content"],
            })
    return {
        "findings": new_findings,           # 会被 operator.add 追加进 state
        "iterations": state["iterations"] + 1,
    }

# ---------- ③ 反思节点 ----------
class Reflection(BaseModel):
    is_sufficient: bool = Field(description="现有资料是否足以写出高质量报告")
    next_query: str = Field(description="若不足，给出下一步最该补充检索的查询；足够则留空")

def reflect_node(state: ResearchState) -> dict:
    reflector = llm.with_structured_output(Reflection)
    digest = "\n".join(f"- {f['title']}: {f['content'][:200]}" for f in state["findings"])
    result = reflector.invoke([
        SystemMessage(content="评估资料是否足够回答研究问题，指出缺口。"),
        HumanMessage(content=f"研究问题：{state['topic']}\n\n已有资料：\n{digest}"),
    ])
    return {"reflection": result.next_query if not result.is_sufficient else ""}

# ---------- ④ 写作节点 ----------
def write_node(state: ResearchState) -> dict:
    sources = "\n".join(
        f"[{i+1}] {f['title']} — {f['url']}\n{f['content'][:500]}"
        for i, f in enumerate(state["findings"])
    )
    report = llm.invoke([
        SystemMessage(content="你是研究报告撰写者。综合资料写一份结构化、带编号引用的报告。"),
        HumanMessage(content=f"研究问题：{state['topic']}\n\n资料来源：\n{sources}"),
    ])
    return {"report": report.content}
```

### 5.3 用图把节点连起来（含循环）

核心在**条件边**：反思之后，根据"是否充足 + 是否到轮数上限"决定回到检索还是去写作。

```python
from langgraph.graph import StateGraph, START, END

def route_after_reflect(state: ResearchState) -> str:
    """条件边：返回值是下一个节点的名字"""
    if state["reflection"] == "":                       # 反思认为足够
        return "write"
    if state["iterations"] >= state["max_iterations"]:  # 到达轮数上限
        return "write"
    return "search"                                     # 否则继续检索（循环！）

builder = StateGraph(ResearchState)
builder.add_node("plan", plan_node)
builder.add_node("search", search_node)
builder.add_node("reflect", reflect_node)
builder.add_node("write", write_node)

builder.add_edge(START, "plan")
builder.add_edge("plan", "search")
builder.add_edge("search", "reflect")
builder.add_conditional_edges("reflect", route_after_reflect, {
    "search": "search",   # 回到检索 → 形成循环
    "write": "write",
})
builder.add_edge("write", END)

# checkpointer 让长任务可中断/恢复、可做人在回路
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())
```

可视化（在 Jupyter 里）：

```python
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

得到的就是 `plan → search → reflect →(条件)→ search/write → END` 这张带回环的图。

### 5.4 运行与流式输出

```python
config = {"configurable": {"thread_id": "research-001"}}   # 配合 checkpointer 标识一次会话

inputs = {
    "topic": "2026 年小型语言模型（SLM）在边缘设备上的部署现状与挑战",
    "max_iterations": 3,
}

# stream 可以逐节点看到中间过程（对调试和给用户反馈都重要）
for event in graph.stream(inputs, config, stream_mode="updates"):
    for node, update in event.items():
        print(f"== 节点 {node} 完成 ==")

final = graph.get_state(config).values
print(final["report"])
```

到这里，你已经拥有一个**会规划、会循环检索、会自我反思、会写带引用报告**的 Deep Research Agent。
而且因为开了 LangSmith，每一步都在云端有 trace 可看。

---

## 6. 用 LangSmith 做可观测、评测与调试

### 6.1 Tracing（零侵入）

只要前面设了 `LANGSMITH_TRACING=true`，运行上面的图后，去 LangSmith 网页就能看到一棵调用树：

```
research-001
├── plan        (claude-opus-4-8, 1.2s, 850 tokens)
├── search      (tavily ×5)
├── reflect     (claude-opus-4-8, 0.9s)   → next_query="..."
├── search      (第二轮)
├── reflect                                → is_sufficient=true
└── write       (claude-opus-4-8, 4.1s, 3200 tokens)
```

点开任意节点能看到**完整的 prompt、模型原始输出、token、耗时**。当 Agent"跑偏"时（比如反思永远说不够、陷入循环），trace 是定位问题最快的手段。

### 6.2 给非 LangChain 代码加追踪

```python
from langsmith import traceable

@traceable(run_type="tool")
def my_custom_scraper(url: str) -> str:
    ...   # 任何普通函数，加了装饰器就会进 trace
```

### 6.3 Evaluation（量化"改得好不好"）

Deep Research 最怕"改了 prompt，感觉变好了，其实变差了"。用 LangSmith 评测把它量化：

```python
from langsmith import Client
client = Client()

# 1) 建数据集：一组研究问题（可选带参考答案）
dataset = client.create_dataset("deep-research-evals")
client.create_examples(
    inputs=[{"topic": "RAG 与长上下文窗口的取舍"},
            {"topic": "2026 年开源向量数据库对比"}],
    dataset_id=dataset.id,
)

# 2) 定义评测器：这里用 "LLM as judge" 给报告打分
from langsmith.evaluation import evaluate

def report_quality(run, example) -> dict:
    report = run.outputs["report"]
    judge = ChatAnthropic(model="claude-opus-4-8")
    score = judge.invoke(
        f"按 0-1 给这份研究报告的'引用充分性'打分，只输出数字：\n{report}"
    )
    return {"key": "citation_score", "score": float(score.content.strip())}

# 3) 跑评测：把整个 graph 当作被测对象
def run_graph(inputs):
    cfg = {"configurable": {"thread_id": "eval"}}
    return graph.invoke({**inputs, "max_iterations": 2}, cfg)

evaluate(run_graph, data="deep-research-evals", evaluators=[report_quality])
```

之后每次改完 Agent，跑一遍评测，就能看到分数曲线，做到"用数据驱动迭代"。

---

## 7. 对比：框架自研 Agent vs OpenClaw / Claude Code

这是本文的重点之一。**LangGraph 写的 Agent** 和 **OpenClaw / Claude Code 这类现成 Agent CLI**，做 research 时是两种完全不同的范式。

### 7.1 它们分别是什么

- **LangChain/LangGraph 路线**：你是**作者**。你用库**自己搭**一个 Agent，掌控每一个节点、每一条边、每一个 prompt。产物是**你的应用代码**。
- **OpenClaw / Claude Code 路线**：你是**用户/操作者**。它们是**已经做好的、通用的自治 Agent**（运行在终端里），自带规划、工具调用、文件读写、shell 执行、循环反思等能力。你通过**自然语言指令**驱动它干活。

> OpenClaw 是开源的、CLI 形态的多智能体框架，自带庞大的"skills"生态；
> Claude Code 是 Anthropic 官方的 CLI / IDE 编码与任务 Agent。两者共同点是：
> **开箱即用的自治 Agent，控制流由它内部决定，你主要靠 prompt 引导。**

### 7.2 核心区别对照表

| 维度 | LangGraph 自研 Agent | OpenClaw / Claude Code |
|---|---|---|
| **你的角色** | 开发者 / 作者，写代码定义 Agent | 操作者，用自然语言下指令 |
| **控制流** | 你显式定义（图的节点与边），完全可控 | Agent 内部自主决定，你看不到/改不动底层图 |
| **定制深度** | 任意定制：每个 prompt、每步逻辑、状态结构 | 靠 prompt、配置、skills 间接影响；底层不可改 |
| **上手成本** | 高：要写代码、懂状态机、调 prompt | 低：装好直接对话即可开始 research |
| **可复现 / 可嵌入** | 强：是确定性程序，可作为服务嵌入产品 | 弱：是交互式会话，难直接嵌进你的后端 |
| **可观测 / 评测** | LangSmith 全链路 trace + 批量评测 | 通常只有会话日志，缺乏结构化评测体系 |
| **多用户 / 规模化** | 天生面向程序化、高并发、API 化部署 | 面向单人/单机交互式使用 |
| **工具范围** | 你给它什么工具就有什么（需自己接） | 自带 shell/文件/网络等强力工具，能力宽 |
| **典型产物** | 一个"研究服务"或产品功能 | 一次"研究会话"的结果（报告/文件） |
| **适合** | 要把 research 能力**产品化、批量化、可控化** | **个人**快速做一次性深度调研、探索性任务 |

### 7.3 用一句话概括

- **OpenClaw / Claude Code** 像是"**雇一个全能研究员**"：你说需求，他自己想办法、自己用工具、自己交报告。**快、强、但过程是黑盒，难嵌入你的系统、难做严格评测**。
- **LangGraph** 像是"**自己设计一条研究流水线**"：每个工位（规划/检索/反思/写作）都是你定义的，**慢一点、但完全可控、可复现、可观测、可规模化部署**。

### 7.4 该怎么选（决策树）

```
你只是想"现在就得到一份深度调研"，自己用？
   └─► 用 Claude Code / OpenClaw，直接对话。最快。

你要把"深度研究"做成产品功能 / API / 给很多用户用 / 要严格评测质量？
   └─► 用 LangGraph 自研 Agent + LangSmith 评测。可控、可复现、可监控。

你要做一次性但很复杂、需要跑代码/操作文件系统的研究探索？
   └─► Claude Code（工具能力强、能写能跑），必要时把产出再接入自研流程。

两者结合：
   └─► 用 Claude Code 快速做原型 / 探索最佳流程 →
       再把验证过的流程用 LangGraph 固化成可部署的产品。  ← 实践中常见且高效
```

> 补充：Claude Code / OpenClaw 这类 Agent，其内部其实**也在做和 LangGraph 类似的事**
> （规划、工具循环、反思），只是把这套控制流**封装好了、藏起来了**。
> 区别不在"谁更智能"，而在**"控制权和可观测性在你手里，还是在 Agent 手里"**。
> 而且无论走哪条路，**LangSmith 都能用上**：自研 Agent 自动接入；
> 对 Claude Code 这类工具，你也可以把它当作"一个外部工具节点"纳入自研流程并加 trace。

---

## 8. 选型建议与常见误区

**何时只用 LangChain（不上图）**
线性 RAG 问答、文本分类/抽取/改写、单轮工具调用——LCEL 一条链足够，别为了用图而用图。

**何时上 LangGraph**
出现「循环、分支、需要记忆、需要人工审批、长任务要能恢复」中的任意一个，就该用图。Deep Research 至少占了循环+分支。

**LangSmith 几乎总该开**
追踪是旁路、零成本、调 Agent 时救命。生产环境再加上评测与监控。

**常见误区**
- ❌ 用早期的 `AgentExecutor` / `initialize_agent` 搭复杂 Agent —— 已是遗留方案，复杂控制流请用 LangGraph。
- ❌ 把 LangChain 当成"必须全家桶" —— 三个库可以单独用，模型也可以直接用厂商 SDK，按需取用。
- ❌ 让循环没有上限 —— 反思型 Agent 一定要有 `max_iterations` 之类的熔断，否则可能无限检索。
- ❌ 凭感觉调 prompt —— 没有 LangSmith 评测数据集，"变好了"只是错觉。
- ❌ 拿 Claude Code 的会话结果直接当成"可复现的生产能力" —— 交互式会话难以严格复现与规模化，产品化要固化成代码。

---

## 9. 附录

### 术语表

| 术语 | 含义 |
|---|---|
| LCEL | LangChain Expression Language，用 `|` 串联组件成链 |
| State | LangGraph 中贯穿全图的共享数据结构 |
| Node / Edge | 图的节点（函数）与边（连接），条件边实现分支/循环 |
| Reducer | 定义 state 字段多次更新如何合并（如 `add_messages`、`operator.add`） |
| Checkpointer | 持久化每步 state，支持中断恢复、时间旅行、人在回路 |
| Trace | LangSmith 记录的一次执行的完整调用树 |
| LLM as judge | 用一个 LLM 给另一个 LLM 的输出打分的评测方法 |
| ReAct | Reason+Act，"推理-行动-观察"循环的经典 Agent 范式 |

### 参考链接（建议以官方最新文档为准）

- LangGraph 文档：https://langchain-ai.github.io/langgraph/
- LangChain 文档：https://python.langchain.com/
- LangSmith 文档：https://docs.smith.langchain.com/
- 官方 Deep Research / Open Deep Research 参考实现（GitHub 上 LangChain 官方仓库有同名示例项目，强烈建议对照阅读）

---

> **小结**：LangChain 给积木，LangGraph 给控制流，LangSmith 给"眼睛"。
> 三者组合让你**自己造**一个可控、可观测、可规模化的 Deep Research Agent；
> 而 OpenClaw / Claude Code 则是**现成的自治研究员**——更快上手，但控制权和可观测性在它们手里。
> 选型的本质就是一句话：**你要"用一个研究 Agent"，还是要"造一个研究 Agent"。**
