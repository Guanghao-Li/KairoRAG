"""Eval 使用的 fake provider 与固定输出。"""

from __future__ import annotations

from typing import Any

from kairorag.eval.datasets import EvalCase


class FakeEvalProviders:
    """提供确定性的检索、Agent、freshness、rerank 和写回结果。"""

    def retrieval_result(self, case: EvalCase) -> dict[str, Any]:
        if case.case_id.endswith("2"):
            retrieved = ["chunk-miss", "chunk-citation"]
            read = ["chunk-citation"]
        else:
            retrieved = ["chunk-rag", "chunk-other"]
            read = ["chunk-rag"]
        return {
            "case_id": case.case_id,
            "expected_chunk_ids": case.expected_chunk_ids,
            "retrieved_chunk_ids": retrieved,
            "read_chunk_ids": read,
            "context_tokens": 320,
        }

    def grounded_answer(self, case: EvalCase) -> dict[str, Any]:
        if not case.expected_chunk_ids:
            return {
                "case_id": case.case_id,
                "answer": "当前没有足够证据，无法确认。",
                "citations": [],
                "read_chunk_ids": [],
                "refused_without_evidence": True,
                "used_search_snippet_as_evidence": False,
            }
        return {
            "case_id": case.case_id,
            "answer": "RAG 结论来自已读取 chunk。",
            "citations": list(case.expected_chunk_ids),
            "read_chunk_ids": list(case.expected_chunk_ids),
            "refused_without_evidence": False,
            "used_search_snippet_as_evidence": False,
        }

    def agent_result(self, case: EvalCase) -> dict[str, Any]:
        invalid = case.case_id.endswith("2")
        return {
            "case_id": case.case_id,
            "planning_success": not invalid,
            "invalid_tool_calls": 1 if invalid else 0,
            "tool_calls": 3,
            "tool_budget_exceeded": False,
            "chunk_read_before_answer": True,
            "final_answer_from_grounded_generator": True,
        }

    def freshness_result(self, case: EvalCase) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "expected_status": case.expected_freshness_status,
            "predicted_status": case.expected_freshness_status,
            "stale_detected": case.expected_freshness_status == "closed",
        }

    def rerank_pair(self, case: EvalCase) -> tuple[dict[str, Any], dict[str, Any]]:
        before = {
            "case_id": case.case_id,
            "expected_chunk_ids": case.expected_chunk_ids,
            "retrieved_chunk_ids": ["chunk-other", *case.expected_chunk_ids],
        }
        after = {
            "case_id": case.case_id,
            "expected_chunk_ids": case.expected_chunk_ids,
            "retrieved_chunk_ids": [*case.expected_chunk_ids, "chunk-other"],
            "latency_ms": 12.0,
        }
        return before, after

    def writeback_result(self, case: EvalCase) -> dict[str, Any]:
        unauthorized = case.case_id.endswith("1")
        return {
            "case_id": case.case_id,
            "requested_apply": unauthorized,
            "authorized": not unauthorized,
            "blocked": unauthorized,
            "dry_run": unauthorized,
            "allowed_field_violation": False,
            "audit_log_created": True,
        }
