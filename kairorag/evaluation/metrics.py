"""Deterministic RAG and job verification metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any


def retrieval_hit_rate(gold_chunk_ids: list[str], search_results: list[str], read_chunks: list[str]) -> float:
    """Whether at least one gold chunk was retrieved or read."""

    if not gold_chunk_ids:
        return 0.0
    seen = set(search_results) | set(read_chunks)
    return 1.0 if set(gold_chunk_ids) & seen else 0.0


def evidence_recall(gold_chunk_ids: list[str], read_chunks: list[str]) -> float:
    """Fraction of gold chunks covered by read chunks."""

    if not gold_chunk_ids:
        return 0.0
    return len(set(gold_chunk_ids) & set(read_chunks)) / len(set(gold_chunk_ids))


def groundedness_score(gold_chunk_ids: list[str], citation_ids: list[str], read_chunks: list[str]) -> float:
    """Score citations that are read and overlap gold evidence."""

    if not citation_ids:
        return 0.0
    citation_set = set(citation_ids)
    read_set = set(read_chunks)
    if not citation_set <= read_set:
        return 0.0
    if gold_chunk_ids and citation_set & set(gold_chunk_ids):
        return 1.0
    return 0.5


def skill_extraction_scores(predicted_skills: list[str], expected_skills: list[str]) -> dict[str, float]:
    """Precision/recall/F1 for normalized skill strings."""

    predicted = {skill.lower() for skill in predicted_skills}
    expected = {skill.lower() for skill in expected_skills}
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rag_metrics(eval_item: dict[str, Any], agent_result: dict[str, Any]) -> dict[str, float]:
    """Compute deterministic metrics for one QA example."""

    search_results: list[str] = []
    for event in agent_result.get("retrieval_trace", []):
        if "chunk_ids" in event.get("outputs", {}):
            search_results.extend(event["outputs"]["chunk_ids"])
    read_chunks = agent_result.get("metrics", {}).get("budget", {}).get("read_chunks", [])
    citation_ids = [citation.get("chunk_id") for citation in agent_result.get("citations", [])]
    skill_scores = skill_extraction_scores(
        agent_result.get("metrics", {}).get("predicted_skills", []),
        eval_item.get("expected_skills", []),
    )
    return {
        "retrieval_hit_rate": retrieval_hit_rate(eval_item.get("gold_chunk_ids", []), search_results, read_chunks),
        "evidence_recall": evidence_recall(eval_item.get("gold_chunk_ids", []), read_chunks),
        "groundedness_score": groundedness_score(eval_item.get("gold_chunk_ids", []), citation_ids, read_chunks),
        "skill_extraction_precision": skill_scores["precision"],
        "skill_extraction_recall": skill_scores["recall"],
        "skill_extraction_f1": skill_scores["f1"],
        "context_token_usage": float(agent_result.get("metrics", {}).get("estimated_context_tokens", 0)),
        "tool_call_count": float(agent_result.get("metrics", {}).get("tool_call_count", 0)),
        "latency_ms": float(agent_result.get("metrics", {}).get("latency_ms", 0)),
    }


def job_verification_metrics(expected: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Compute job verification metrics."""

    by_id = {item["job_id"]: item for item in predictions}
    total = len(expected)
    correct = 0
    active_tp = 0
    active_fp = 0
    closed_tp = 0
    closed_fn = 0
    stale_count = 0
    update_correct = 0
    domain_match = 0
    reindex_correct = 0
    for item in expected:
        pred = by_id.get(item["job_id"], {})
        status = pred.get("new_status")
        if status == item.get("expected_status"):
            correct += 1
        if status == "active":
            if item.get("expected_status") == "active":
                active_tp += 1
            else:
                active_fp += 1
        if item.get("expected_status") == "closed":
            if status == "closed":
                closed_tp += 1
            else:
                closed_fn += 1
        if status == "stale":
            stale_count += 1
        if item.get("expected_status") == "updated":
            update_correct += 1 if status == "updated" and bool(pred.get("updated_fields")) else 0
        else:
            update_correct += 1
        evidence_blob = str(pred.get("evidence", "")).lower()
        if item.get("expected_evidence_domain", "") in evidence_blob:
            domain_match += 1
        if bool(pred.get("needs_reindex")) == bool(item.get("expected_needs_reindex")):
            reindex_correct += 1
    return {
        "verification_accuracy": correct / total if total else 0.0,
        "active_precision": active_tp / (active_tp + active_fp) if active_tp + active_fp else 0.0,
        "closed_recall": closed_tp / (closed_tp + closed_fn) if closed_tp + closed_fn else 0.0,
        "stale_rate": stale_count / total if total else 0.0,
        "update_correctness": update_correct / total if total else 0.0,
        "evidence_domain_match": domain_match / total if total else 0.0,
        "reindex_trigger_accuracy": reindex_correct / total if total else 0.0,
    }


def average_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: sum(item.get(key, 0.0) for item in items) / len(items) for key in keys}

