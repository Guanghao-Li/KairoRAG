"""Legacy / Deprecated：离线 rule-based planner，后续会被 LLM agent planner 替换。"""

from __future__ import annotations

from dataclasses import dataclass


VERIFY_TERMS = [
    "还在招",
    "还能投",
    "正在招聘",
    "已经关闭",
    "更新",
    "剔除",
    "验证",
    "active",
    "closed",
    "fresh",
]

KEYWORD_HINTS = [
    "rag",
    "langgraph",
    "mcp",
    "akashic-agent",
    "northstar",
    "quietcloud",
    "civicflow",
    "deltaops",
    "公司",
    "岗位",
]


@dataclass
class Plan:
    search_tool: str
    requires_verification: bool
    rationale: str


def plan_query(question: str, verify_jobs: bool = False) -> Plan:
    """Legacy：基于规则选择检索路径。"""

    lowered = question.lower()
    requires_verification = verify_jobs or any(term in lowered for term in VERIFY_TERMS)
    has_keyword = any(term in lowered for term in KEYWORD_HINTS)
    if requires_verification:
        return Plan("hybrid_search", True, "Question asks for current job status or freshness.")
    if has_keyword:
        return Plan("hybrid_search", False, "Question contains exact company, project, or skill terms.")
    return Plan("semantic_search", False, "Question is mostly conceptual, so semantic retrieval is preferred.")
