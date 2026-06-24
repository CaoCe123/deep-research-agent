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
