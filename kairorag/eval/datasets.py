"""Eval 数据结构与内置离线 fixture。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """单条 eval case。"""

    case_id: str
    query: str
    expected_chunk_ids: list[str]
    expected_answer_contains: list[str]
    expected_skills: list[str]
    expected_freshness_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCaseResult:
    """单条 eval case 的执行结果。"""

    case_id: str
    query: str
    passed: bool
    metrics: dict[str, Any]
    errors: list[str]
    trace_path: str | None = None


@dataclass(frozen=True)
class EvalSuiteResult:
    """一个 eval suite 的聚合结果。"""

    suite: str
    case_count: int
    passed_count: int
    failed_count: int
    aggregate_metrics: dict[str, Any]
    cases: list[EvalCaseResult]


def load_fixture_cases(suite: str) -> list[EvalCase]:
    """返回内置离线 fixture，不访问外部服务。"""

    fixtures = {
        "retrieval": [
            EvalCase("retrieval-1", "哪些岗位要求 RAG？", ["chunk-rag"], ["RAG"], ["retrieval"]),
            EvalCase("retrieval-2", "哪些文档提到 citations？", ["chunk-citation"], ["citation"], ["chunk_read"]),
        ],
        "groundedness": [
            EvalCase("grounded-1", "总结证据", ["chunk-rag"], ["RAG"], ["generate_grounded_answer"]),
            EvalCase("grounded-2", "无证据时怎么办？", [], ["无法确认"], ["generate_grounded_answer"]),
        ],
        "agent": [
            EvalCase("agent-1", "读取 chunk 后回答", ["chunk-rag"], ["RAG"], ["hybrid_search", "chunk_read"]),
            EvalCase("agent-2", "未知工具应失败", ["chunk-safe"], ["安全拒绝"], ["chunk_read"]),
        ],
        "freshness": [
            EvalCase(
                "freshness-1",
                "岗位还开放吗？",
                ["job-active"],
                ["active"],
                ["verify_job_freshness"],
                expected_freshness_status="active",
            ),
            EvalCase(
                "freshness-2",
                "关闭岗位还能推荐吗？",
                ["job-closed"],
                ["closed"],
                ["verify_job_freshness"],
                expected_freshness_status="closed",
            ),
        ],
        "reranker": [
            EvalCase("reranker-1", "RAG 相关结果排序", ["chunk-rag"], ["RAG"], ["hybrid_search"]),
            EvalCase("reranker-2", "freshness 相关结果排序", ["job-active"], ["active"], ["hybrid_search"]),
        ],
        "writeback": [
            EvalCase("writeback-1", "未授权写回必须阻断", ["job-closed"], ["dry-run"], ["apply_freshness_update"]),
            EvalCase("writeback-2", "允许字段写回", ["job-active"], ["audit"], ["apply_freshness_update"]),
        ],
    }
    return list(fixtures.get(suite, []))
