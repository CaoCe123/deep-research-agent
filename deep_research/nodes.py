from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from . import search as search_pkg
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
    search_fn = search_pkg.get_search_fn(state["search_source"])
    queries = state["sub_questions"] if state["iterations"] == 0 else [state["reflection"]]
    new_findings: list[dict] = []
    for q in queries:
        for hit in search_fn(q):
            hit = dict(hit)
            hit["content"] = _summarize(hit["content"]) if hit["content"] else ""
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


def _format_references(findings: list[dict]) -> str:
    lines = ["## 参考文献"]
    for i, f in enumerate(findings, 1):
        if f.get("doi") or f.get("authors"):           # academic source
            names = f.get("authors") or []
            authors = ", ".join(names[:3]) + (" 等" if len(names) > 3 else "")
            parts = [p for p in [authors, f.get("title"),
                                 f.get("venue"), str(f["year"]) if f.get("year") else ""] if p]
            ref = ". ".join(parts)
            if f.get("doi"):
                ref += f". DOI:{f['doi']}"
            cite = f"（被引 {f['cited_by']} 次）" if f.get("cited_by") is not None else ""
            lines.append(f"[{i}] {ref} {cite}".rstrip())
        else:                                          # web source fallback
            lines.append(f"[{i}] {f.get('title', '')} — {f.get('url', '')}")
    return "\n".join(lines)


def write_node(state: ResearchState) -> dict:
    findings = state["findings"]
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
