"""Cloud LLM Agent 的状态、trace 和结果 schema。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kairorag.cloud.answering import CloudCitation
from kairorag.cloud.freshness.schemas import JobFreshnessResult
from kairorag.cloud.retriever import CloudReadChunk, CloudSearchResult


@dataclass(frozen=True)
class AgentToolInput:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentToolOutput:
    name: str
    ok: bool
    result: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class AgentTraceStep:
    step: str
    detail: dict[str, Any]


def default_agent_metrics() -> dict[str, Any]:
    return {
        "tool_call_count": 0,
        "invalid_tool_call_count": 0,
        "blocked_final_answer_count": 0,
        "read_chunk_count": 0,
        "search_result_count": 0,
        "final_citation_count": 0,
        "freshness_verification_count": 0,
        "freshness_update_count": 0,
        "freshness_update_applied_count": 0,
        "max_tool_calls_reached": False,
        "llm_planning_rounds": 0,
    }


@dataclass
class AgentState:
    query: str
    search_results: dict[str, CloudSearchResult] = field(default_factory=dict)
    read_chunks: dict[str, CloudReadChunk] = field(default_factory=dict)
    trace: list[AgentTraceStep] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=default_agent_metrics)
    tool_call_count: int = 0
    invalid_tool_call_count: int = 0
    blocked_final_answer_count: int = 0
    finished: bool = False
    final_answer: str = ""
    final_citations: list[CloudCitation] = field(default_factory=list)
    freshness_results: dict[str, JobFreshnessResult] = field(default_factory=dict)
    filters: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentRunResult:
    query: str
    answer: str
    citations: list[CloudCitation]
    trace: list[AgentTraceStep]
    metrics: dict[str, Any]
    verification_results: list[JobFreshnessResult] = field(default_factory=list)
