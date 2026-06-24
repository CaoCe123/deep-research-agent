import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]  # reducer: accumulate across rounds
    reflection: str
    iterations: int
    max_iterations: int
    search_source: str
    report: str


class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3-5 个互补、可独立检索的子问题")


class Reflection(BaseModel):
    is_sufficient: bool = Field(description="现有资料是否足以写出高质量报告")
    next_query: str = Field(description="若不足，下一步最该补充检索的查询；足够则留空字符串")
