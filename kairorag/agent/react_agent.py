"""Rule-based ReAct-style Agentic RAG loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kairorag.agent.answer_generator import generate_answer
from kairorag.agent.planner import plan_query
from kairorag.config import INDEX_DIR, KairoConfig, RAW_DATA_DIR
from kairorag.indexing.build_index import build_indexes
from kairorag.ingestion.jd_loader import load_jobs
from kairorag.maintenance.freshness import needs_verification
from kairorag.maintenance.index_refresher import refresh_index_for_jobs
from kairorag.maintenance.job_verifier import verify_job
from kairorag.maintenance.knowledge_base_updater import apply_verification_result
from kairorag.retrieval.chunk_read import chunk_read
from kairorag.retrieval.context_budget import ContextBudgetManager
from kairorag.retrieval.context_compressor import compress_context
from kairorag.retrieval.hybrid_search import hybrid_search
from kairorag.retrieval.keyword_search import keyword_search
from kairorag.retrieval.reranker import rerank_context
from kairorag.retrieval.semantic_search import semantic_search
from kairorag.schemas import JobVerificationResult, ReadChunkResult


def ensure_indexes(index_dir: str | Path = INDEX_DIR) -> None:
    """Build local indexes if they are missing."""

    index_path = Path(index_dir)
    if not (index_path / "keyword_index.pkl").exists() or not (index_path / "vector_store.pkl").exists():
        build_indexes(RAW_DATA_DIR, index_path)


class ReactAgent:
    """Offline ReAct-style loop with explicit retrieval actions."""

    def __init__(
        self,
        index_dir: str | Path = INDEX_DIR,
        max_tool_calls: int = KairoConfig.max_tool_calls,
        max_chunks_to_read: int = KairoConfig.max_chunks_to_read,
        max_context_tokens: int = KairoConfig.max_context_tokens,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.max_tool_calls = max_tool_calls
        self.max_chunks_to_read = max_chunks_to_read
        self.max_context_tokens = max_context_tokens

    def run(
        self,
        question: str,
        verify_jobs: bool = False,
        apply_kb_updates: bool = False,
    ) -> dict[str, Any]:
        """Execute search, read, optional verification, and grounded answer generation."""

        ensure_indexes(self.index_dir)
        start = time.perf_counter()
        plan = plan_query(question, verify_jobs=verify_jobs)
        trace: list[dict[str, Any]] = [
            {"step": 1, "tool": "planner", "inputs": {"question": question}, "outputs": plan.__dict__}
        ]
        budget = ContextBudgetManager(
            max_search_results=KairoConfig.max_search_results,
            max_chunks_to_read=self.max_chunks_to_read,
            max_context_tokens=self.max_context_tokens,
            max_tool_calls=self.max_tool_calls,
        )

        budget.record_tool_call(plan.search_tool)
        if plan.search_tool == "keyword_search":
            results = keyword_search(question, top_k=budget.max_search_results, index_dir=self.index_dir)
        elif plan.search_tool == "semantic_search":
            results = semantic_search(question, top_k=budget.max_search_results, index_dir=self.index_dir)
        else:
            results = hybrid_search(question, top_k=budget.max_search_results, index_dir=self.index_dir)
        results = budget.limit_search_results(results)
        trace.append(
            {
                "step": len(trace) + 1,
                "tool": plan.search_tool,
                "inputs": {"query": question},
                "outputs": {"chunk_ids": [result.chunk_id for result in results]},
            }
        )

        budget.record_tool_call("rerank_context")
        reranked = rerank_context(question, results, top_k=self.max_chunks_to_read)
        trace.append(
            {
                "step": len(trace) + 1,
                "tool": "rerank_context",
                "inputs": {"candidate_count": len(results)},
                "outputs": {"chunk_ids": [result.chunk_id for result in reranked]},
            }
        )

        read_chunks: list[ReadChunkResult] = []
        for result in reranked:
            if not budget.can_call_tool() or not budget.can_read(result.chunk_id):
                continue
            budget.record_tool_call("chunk_read")
            chunk = chunk_read(result.chunk_id, window=1, index_dir=self.index_dir)
            if chunk.text and budget.record_read(chunk):
                read_chunks.append(chunk)
            trace.append(
                {
                    "step": len(trace) + 1,
                    "tool": "chunk_read",
                    "inputs": {"chunk_id": result.chunk_id, "window": 1},
                    "outputs": {"has_text": bool(chunk.text), "title": chunk.metadata.get("title", "")},
                }
            )

        budget.record_tool_call("compress_context")
        compressed = compress_context(question, read_chunks)
        trace.append(
            {
                "step": len(trace) + 1,
                "tool": "compress_context",
                "inputs": {"read_chunk_count": len(read_chunks)},
                "outputs": {"block_count": len(compressed)},
            }
        )

        verification_results: list[JobVerificationResult] = []
        if plan.requires_verification:
            jobs_by_id = {job.job_id: job for job in load_jobs(RAW_DATA_DIR / "jobs.csv")}
            candidate_job_ids = []
            for chunk in read_chunks:
                if chunk.metadata.get("source_type") == "job":
                    job_id = str(chunk.metadata.get("job_id", chunk.metadata.get("doc_id", "")))
                    if job_id and job_id not in candidate_job_ids:
                        candidate_job_ids.append(job_id)
            for job_id in candidate_job_ids:
                if not budget.can_call_tool():
                    break
                job = jobs_by_id.get(job_id)
                if not job:
                    continue
                if not needs_verification(
                    job.last_verified_at,
                    job.verification_status,
                    explicit=verify_jobs or plan.requires_verification,
                ):
                    continue
                budget.record_tool_call("verify_job_status_tool")
                result = verify_job(job, search=True)
                verification_results.append(result)
                trace.append(
                    {
                        "step": len(trace) + 1,
                        "tool": "verify_job_status_tool",
                        "inputs": {"job_id": job_id},
                        "outputs": result.to_dict(),
                    }
                )
                if not budget.can_call_tool():
                    break
                budget.record_tool_call("update_knowledge_base_tool")
                report = apply_verification_result(result, apply=apply_kb_updates)
                trace.append(
                    {
                        "step": len(trace) + 1,
                        "tool": "update_knowledge_base_tool",
                        "inputs": {"job_id": job_id, "apply": apply_kb_updates},
                        "outputs": {
                            "action": report.action,
                            "changed_fields": report.changed_fields,
                            "needs_reindex": report.needs_reindex,
                        },
                    }
                )
                if apply_kb_updates and report.needs_reindex:
                    refresh = refresh_index_for_jobs([job_id])
                    trace.append(
                        {
                            "step": len(trace) + 1,
                            "tool": "refresh_index_tool",
                            "inputs": {"job_ids": [job_id]},
                            "outputs": refresh.to_dict(),
                        }
                    )

        metrics = {
            "tool_call_count": budget.tool_call_count,
            "read_chunk_count": len(read_chunks),
            "estimated_context_tokens": budget.estimated_context_tokens,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "budget": budget.report().to_dict(),
        }
        return generate_answer(
            question,
            read_chunks,
            compressed,
            trace,
            verification_results=verification_results,
            metrics=metrics,
        )
