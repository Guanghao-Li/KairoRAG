"""Batch evaluators for QA and job verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairorag.agent.react_agent import ReactAgent
from kairorag.config import RAW_DATA_DIR, RESULTS_DIR
from kairorag.evaluation.dataset import load_job_verification_eval, load_qa_eval
from kairorag.evaluation.metrics import average_metric_dict, job_verification_metrics, rag_metrics
from kairorag.evaluation.report import write_json_report, write_markdown_report
from kairorag.ingestion.jd_loader import load_jobs
from kairorag.maintenance.job_verifier import verify_job


def run_qa_eval(eval_file: str | Path, output: str | Path, include_traces: bool = False) -> dict[str, Any]:
    """Run deterministic RAG eval."""

    items = load_qa_eval(eval_file)
    agent = ReactAgent()
    cases = []
    metric_items = []
    for item in items:
        result = agent.run(item["question"])
        metrics = rag_metrics(item, result)
        metric_items.append(metrics)
        retrieval_path = [event["tool"] for event in result.get("retrieval_trace", [])]
        read_chunks = result.get("metrics", {}).get("budget", {}).get("read_chunks", [])
        case = {
            "id": item.get("id"),
            "question": item.get("question"),
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "read_chunks": read_chunks,
            "retrieval_path": retrieval_path,
            "metrics": metrics,
        }
        if include_traces:
            case["trace"] = result.get("retrieval_trace", [])
        cases.append(case)
    payload = {"overall_metrics": average_metric_dict(metric_items), "cases": cases}
    output_path = Path(output)
    write_json_report(output_path, payload)
    write_markdown_report(output_path.with_suffix(".md"), payload, "KairoRAG RAG 评测报告")
    return payload


def run_job_verification_eval(
    eval_file: str | Path = Path("data/eval/job_verification_eval.json"),
    output: str | Path = RESULTS_DIR / "job_verification_eval_report.json",
) -> dict[str, Any]:
    """Run deterministic job verification eval."""

    items = load_job_verification_eval(eval_file)
    jobs_by_id = {job.job_id: job for job in load_jobs(RAW_DATA_DIR / "jobs.csv")}
    predictions = []
    cases = []
    for item in items:
        result = verify_job(jobs_by_id[item["job_id"]], search=True)
        prediction = result.to_dict()
        predictions.append(prediction)
        cases.append(
            {
                "id": item.get("id"),
                "job_id": item.get("job_id"),
                "prediction": result.new_status,
                "metrics": {"confidence": result.confidence, "needs_reindex": result.needs_reindex},
                "retrieval_path": ["parse_job_page", "search_web" if len(result.evidence) > 1 else "original_url"],
            }
        )
    payload = {"overall_metrics": job_verification_metrics(items, predictions), "cases": cases}
    output_path = Path(output)
    write_json_report(output_path, payload)
    write_markdown_report(output_path.with_suffix(".md"), payload, "KairoRAG 岗位验证评测报告")
    return payload
